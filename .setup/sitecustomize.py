import sys
import os
import linecache


def excepthook(exc_type, exc_value, tb):
    import traceback
    traceback.print_exception(exc_type, exc_value, tb)

    # walk to the innermost frame (where the exception actually occurred)
    while tb.tb_next:
        tb = tb.tb_next
    frame = tb.tb_frame

    file_name = frame.f_code.co_filename
    line_number = frame.f_lineno
    function_name = frame.f_code.co_name

    # get actual code line from the file using linecache
    line = linecache.getline(file_name, line_number).strip()
    print(f"\n --> Crash at {file_name}:{line_number} in {function_name}\n {line}")

    mode = os.environ.get("PDB_DEBUGGER_MODE", "pdb")

    if mode == "embed":
        # start an interactive IPython shell with the local variables of the frame
        from IPython import embed
        embed(user_ns={**frame.f_globals, **frame.f_locals})
    elif mode == "ipdb":
        from IPython.terminal.debugger import TerminalPdb
        TerminalPdb().interaction(None, tb)
    else:
        # start the standard Python debugger (pdb) with the local variables of the frame
        import pdb
        pdb.post_mortem(tb)

sys.excepthook = excepthook
