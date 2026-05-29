import importlib


REQUIRED_PACKAGES = [
    ("earthengine-api", "ee"),
    ("geemap", "geemap"),
    ("folium", "folium"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("scikit-learn", "sklearn"),
    ("jupyter", "jupyter"),
    ("notebook", "notebook"),
]

OPTIONAL_PACKAGES = [
    ("xgboost", "xgboost"),
    ("tensorflow", "tensorflow"),
]


def check_imports() -> bool:
    print("Checking Python package imports...\n")
    all_ok = True

    for package_name, module_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module_name)
            print(f"[OK] {package_name}")
        except Exception as exc:
            all_ok = False
            print(f"[MISSING] {package_name} -> {exc}")

    return all_ok


def check_earth_engine() -> None:
    try:
        import ee

        ee.Initialize()
        print("\n[OK] Earth Engine initialized successfully.")
    except Exception as exc:
        print("\n[WARN] Earth Engine is not ready in this environment.")
        print(f"       Details: {exc}")
        print("       You may need to run ee.Authenticate() first.")


def check_optional_packages() -> None:
    print("\nOptional packages:")
    for package_name, module_name in OPTIONAL_PACKAGES:
        try:
            importlib.import_module(module_name)
            print(f"[OK] {package_name}")
        except Exception as exc:
            print(f"[OPTIONAL-MISSING] {package_name} -> {exc}")


def main() -> None:
    all_ok = check_imports()
    check_earth_engine()
    check_optional_packages()

    print("\nSummary:")
    if all_ok:
        print("Base dependencies are available.")
    else:
        print("Some dependencies are missing. Install them with pip install -r requirements.txt")


if __name__ == "__main__":
    main()
