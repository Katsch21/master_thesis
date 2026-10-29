SITE=/data/dust/user/hergesk/venvs/mltorch/lib/python3.9/site-packages

for d in "$SITE"/nvidia/*/lib; do
    if [ -d "$d" ]; then
        case ":$LD_LIBRARY_PATH:" in
            *":$d:"*) ;;
            *) export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac
    fi
done
