mở powershell của vscode lên và dùng lệnh Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
sau đó tắt đi mở lại powershell, nhập .venv\Scripts\Activate.ps1
sau đó cài đặt thư viện bằng pip install streamlit opencv-python-headless numpy tqdm torch torchvision ptlflow imageio imageio-ffmpeg
để thực thi chương trình sử dụng lệnh streamlit run app.py
