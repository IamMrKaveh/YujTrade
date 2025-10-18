import os

def save_python_files(root_path: str):
    ignore_dirs = {"Main", "models", "migrations", "__pycache__", "copy-file"}
    ignore_files = {"__init__.py"}

    # مسیر پوشه خروجی (در کنار اسکریپت)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    export_dir = os.path.join(script_dir, "code_exports")
    os.makedirs(export_dir, exist_ok=True)

    for folder, dirs, files in os.walk(root_path):
        # حذف پوشه‌های غیرمجاز
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        # فیلتر فایل‌های مجاز
        py_files = [f for f in files if f.endswith(".py") and f not in ignore_files]
        if not py_files:
            continue

        # مسیر نسبی برای نام فایل خروجی
        relative_folder = os.path.relpath(folder, root_path)
        if relative_folder == ".":
            relative_folder = "root"
        relative_folder = relative_folder.replace("\\", "_").replace("/", "_")

        output_file = os.path.join(export_dir, f"{relative_folder}_code.txt")

        with open(output_file, "w", encoding="utf-8") as out:
            for file in py_files:
                file_path = os.path.join(folder, file)
                relative_path = os.path.relpath(file_path, root_path)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    content = f"--- Error reading file: {e} ---"

                out.write(f"[{relative_path}] :\n\n")
                out.write(content)
                out.write("\n\n=====\n\n")

        print(f"✅ Saved: {os.path.basename(output_file)}")

    print(f"\n🎯 Done! All folders processed successfully.")
    print(f"📂 Output directory: {export_dir}")


if __name__ == "__main__":
    project_path = r"D:\source\Projects\My Projects\Full Projects\Python\YujTrade"
    save_python_files(project_path)
