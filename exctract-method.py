import ast
import sys

def extract_functions(python_file_path, output_txt_path):
    """
    توابع موجود در یک فایل پایتون را استخراج و در فایل متنی ذخیره می‌کند.
    
    پارامترها:
        python_file_path (str): مسیر فایل پایتون ورودی
        output_txt_path (str): مسیر فایل متنی خروجی
    """
    try:
        with open(python_file_path, 'r', encoding='utf-8') as file:
            code = file.read()
        
        # تجزیه کد به درخت نحو انتزاعی
        tree = ast.parse(code)
        
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # استخراج اطلاعات تابع
                func_name = node.name
                args = [arg.arg for arg in node.args.args]
                docstring = ast.get_docstring(node) or "بدون توضیحات"
                
                functions.append({
                    'name': func_name,
                    'args': args,
                    'docstring': docstring
                })
        
        # ذخیره توابع در فایل متنی
        with open(output_txt_path, 'w', encoding='utf-8') as out_file:
            out_file.write(f"توابع استخراج شده از فایل: {python_file_path}\n")
            out_file.write("="*10 + "\n\n")
            
            for func in functions:
                out_file.write(f"نام تابع: {func['name']}\n")
                out_file.write(f"پارامترها: {', '.join(func['args'])}\n")
                out_file.write(f"توضیحات: {func['docstring']}\n")
                out_file.write("\n\n")
                out_file.write("-"*10 + "\n\n")
                
        print(f"توابع با موفقیت استخراج و در {output_txt_path} ذخیره شدند.")
        
    except FileNotFoundError:
        print(f"خطا: فایل {python_file_path} یافت نشد.")
    except Exception as e:
        print(f"خطا در پردازش فایل: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("first argument should be the Python file path and second argument should be the output text file path.")
        sys.exit(1)
    
    python_file = sys.argv[1]
    output_file = sys.argv[2]
    extract_functions(python_file, output_file)