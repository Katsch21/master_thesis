from collections import deque





        # ----
        ### training loop
        # ----
        logger_inst.info("Start training loop")
        last_losses = deque([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], maxlen=10)
        for current_iteration in range(full_config.training_config.max_train_iteration):
            batch_result = training_loop(
                model_inst=model_inst,
                monitor=training_monitor_inst,
                kind_of_data=mode_batch,
                loss_fn=training_loss_inst,
                sampler=training_sampler,
                device=DEVICE,
                sample_columns=full_config.sampler_config.sample_attributes,
                scheduler_handler_inst=scheduler_handler_inst,
                optimizer_inst=optimizer_inst,
            )

            if batch_composition_history_inst is not None:
                batch_composition_history_inst.record(
                    step=current_iteration, cursors=training_sampler.cursors
                )

            # ----
            # Verbose and Metrics that are triggered often
            # ----
            if current_iteration % full_config.record_config.verbose_interval == 0:
                tensorboard_writer.log_lr(optimizer_inst, current_iteration)
                batch_loss = batch_result["loss"].item()
                tensorboard_writer.log_loss(
                    {"batch_loss": batch_loss}, step=current_iteration
                )
                current_lr = optimizer_inst.param_groups[0]["lr"]
                logger_inst.training(
                    f"T-It: {current_iteration} - LR: {current_lr} - batch loss: {batch_loss:.2E}"
                )

            # ----
            #### Evaluation of training and validation data, logging and checkpointing
            # ----
            evaluation_condition = (current_iteration % full_config.record_config.validation_interval == 0)
            if evaluation_condition:
                # evaluation of training data
                logger_inst.info(
                    f"Iteration {current_iteration}. Start evaluation of training data."
                )

                evaluation_training_result = validation_loop(
                    model_inst=model_inst,
                    monitor=training_monitor_inst,
                    kind_of_data=mode_eval_training,
                    loss_fn_inst=validation_loss_inst,
                    sampler_inst=training_sampler,
                    sample_columns=full_config.sampler_config.sample_attributes,
                    device=DEVICE,
                )
                # evaluation of validation
                logger_inst.info(
                    f"Iteration {current_iteration}. Start evaluation of validation data."
                )

                evaluation_validation_result = validation_loop(
                    model_inst=model_inst,
                    monitor=training_monitor_inst,
                    kind_of_data=mode_eval_validation,
                    loss_fn_inst=validation_loss_inst,
                    sampler_inst=validation_sampler,
                    sample_columns=full_config.sampler_config.sample_attributes,
                    device=DEVICE,
                )

                eval_t_loss = evaluation_training_result["loss"].item()
                eval_v_loss = evaluation_validation_result["loss"].item()
                logger_inst.training(
                    f"Iteration: {current_iteration} - TLoss: {eval_t_loss:.2E} VLoss: {eval_v_loss:.2E}"
                )

                # TODO when edges should be tracked add this in a way that is universal and does not break for models without binning layer, e.g. add property to model that returns None if no binning layer is present and add check in log_metrics
                if full_config.record_config.log_metrics:
                    # --- plots on evaluation, on training data ---
                    model_evaluation_state = model_inst.evaluation_state()

                    shared_eval_context_meta_data = {
                        "model_evaluation_state": model_evaluation_state,
                        "target_map": full_config.dataset_config.target_map,
                        "global_step": current_iteration,
                        "default_n_bins": full_config.binning_config.num_bins,
                    }
                    if batch_composition_history_inst is not None:
                        shared_eval_context_meta_data["batch_composition_history"] = batch_composition_history_inst


                    ctx_batch = EvalContext(
                        mode="batch",
                        predictions=batch_result["predictions"],
                        targets=batch_result["targets"],
                        event_weights=batch_result["event_weights"],
                        **shared_eval_context_meta_data,
                    )

                    ctx_train = EvalContext(
                        mode="training",
                        predictions=evaluation_training_result["predictions"],
                        targets=evaluation_training_result["targets"],
                        event_weights=evaluation_training_result["event_weights"],
                        **shared_eval_context_meta_data,
                    )

                    ctx_validation = EvalContext(
                        mode="validation",
                        predictions=evaluation_validation_result["predictions"],
                        targets=evaluation_validation_result["targets"],
                        event_weights=evaluation_validation_result["event_weights"],
                        **shared_eval_context_meta_data,
                    )
                    # ctx_batch.add_feature("kernels", model_inst.kernels)
                    # ctx_batch.add_feature("binning_fn", model_inst.binning_fn)

                    ctx_batch.add_features(
                        *training_monitor_inst.get_plot_gradients(mode_batch),
                        *training_monitor_inst.get_plot_tensors(mode_batch),
                    )

                    ctx_train.add_features(
                        *training_monitor_inst.get_plot_tensors(mode_eval_training),
                        ("loss", eval_t_loss),
                    )

                    ctx_validation.add_features(
                        *training_monitor_inst.get_plot_tensors(mode_eval_validation),
                        ("loss", eval_v_loss),
                    )

                    # --- Plotting
                    evaluation_runner_inst.run_plots(
                        ctx_batch,
                        plots=[
                            # "kernels_monitor",
                            "batch_composition_history",
                        ],
                    )
                    # run metrics and store them
                    evaluation_runner_inst.run_plots(
                        ctx_train,
                        plots=[
                            "confusion_matrix",
                            # "roc", # roc computation is the long factor (14s)
                            "output_score_hh_node",
                            "output_score_hh_node_untransformed",
                            "kernels_monitor",
                            "score_correlation_matrix",
                            # "precision_recall", # deactivated takes 40s to use
                        ],
                    )

                    evaluation_runner_inst.run_plots(
                        ctx_validation,
                        plots=[
                            "confusion_matrix",
                            # "roc", # roc computation is the long factor (4s)
                            "output_score_hh_node",
                            "output_score_hh_node_untransformed",
                            "kernels_monitor",
                            "score_correlation_matrix",
                            # "precision_recall", # takes to long
                        ],
                    )

                    evaluation_runner_inst.run_scalars(
                        ctx=ctx_train,
                        artifact_names={
                            "CrossEntropy/Evaluation Training": "cross_entropy",
                            "Loss/Evaluation Training Loss": "loss",
                        },
                    )

                    evaluation_runner_inst.run_scalars(
                        ctx=ctx_validation,
                        artifact_names={
                            "CrossEntropy/Evaluation Validation": "cross_entropy",
                            "Loss/Validation VLoss": "loss",
                        },
                    )

                last_losses.append(eval_v_loss.cpu().item())
                mean_last_losses = np.mean(last_losses)

                ### checkpoint criteria checks and saving
                if checkpoint_inst.check_criteria(mean_last_losses):
                    checkpoint_inst.create_checkpoint(
                        model=model_inst,
                        optimizer=optimizer_inst,
                        scheduler=scheduler_inst,
                        current_iteration=current_iteration,
                        full_config=full_config,
                    )

                scheduler_handler_inst.step(
                    model_inst, optimizer_inst, metric=eval_v_loss
                )

        from IPython import embed

        embed(header="Training ends: Check if everything is as you thought it would be")
