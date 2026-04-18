import sys

def check_python():
    v = sys.version_info
    if v.major == 3 and v.minor in (10, 11):
        print(f"✅ Python {v.major}.{v.minor} - OK")
        return True
    print(f"⚠️ Python {v.major}.{v.minor} - Recommended 3.10 or 3.11")
    return False

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            return True
        print("⚠️ CUDA not available (CPU mode)")
        return False
    except Exception as e:
        print("❌ PyTorch not installed or failed to import:", e)
        return False

def check_packages():
    required = ["ultralytics", "cv2", "numpy", "pandas", "matplotlib"]
    ok = True
    for pkg in required:
        try:
            __import__(pkg if pkg != "cv2" else "cv2")
            print(f"✅ {pkg} installed")
        except Exception:
            print(f"❌ {pkg} NOT installed")
            ok = False
    return ok

if __name__ == "__main__":
    print("=" * 60)
    print("SETUP VERIFICATION")
    print("=" * 60)
    py_ok = check_python()
    gpu_ok = check_gpu()
    pk_ok = check_packages()

    print("=" * 60)
    if py_ok and pk_ok:
        print("✅ ALL SYSTEMS READY!")
    else:
        print("⚠️ SETUP INCOMPLETE - fix issues above")
    print("=" * 60)
