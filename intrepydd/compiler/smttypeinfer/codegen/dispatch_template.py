def {ORIGINAL_NAME}(*args):
    try:
        __func_names = get_mangled_function_names('{ORIGINAL_NAME}', args)
        __f = None
        for __func_name in __func_names:
            __f = getattr({CPP_MODULE_NAME}, __func_name, None)
            if __f is not None:
                break
        if __f is None:
            __f = {PYTHON_MODULE_NAME}.{ORIGINAL_NAME}
    except UnsupportedType:
        __f = {PYTHON_MODULE_NAME}.{ORIGINAL_NAME}

{EXTRA}

    return __f(*args)
