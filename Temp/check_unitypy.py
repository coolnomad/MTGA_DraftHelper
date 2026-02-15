try:
    import UnityPy
    print('UnityPy version', UnityPy.__version__)
except Exception as e:
    print('UnityPy import failed:', e)
