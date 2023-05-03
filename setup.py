import subprocess

def install_packages(packages):
    for package in packages:
        subprocess.check_call(["pip", "install", package])

if __name__ == "__main__":
    packages_to_install = ["openai", "PyPDF2", "langchain", "pandas"]
    install_packages(packages_to_install)