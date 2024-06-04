- To build binaries of backend you must install all the dependencies globally along with pyinstaller.

```bash
pip install -r requirements.txt
pip install pyinstaller
```

- To build the binaries, run the following command in the backend directory where app is located.

```bash
 pyinstaller  --hidden-import=flask --hidden-import=opencv-python --hidden-import=numpy --hidden-import=onnxruntime --hidden-import=flask_cors app.py
```


