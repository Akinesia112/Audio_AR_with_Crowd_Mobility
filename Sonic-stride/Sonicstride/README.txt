

1. 需要啟用這個資料夾的環境再執行python
	

	啟用虛擬環境。在命令提示字元或 PowerShell 中輸入以下命令：

	WEBCAMGUI\Scripts\activate

	按 Enter 鍵運行此命令。它將啟用名為“WEBCAMGUI”的虛擬環境。您現在可以在該虛擬環境中安裝 Python 包，並運行您的 Python 代碼。


----


2. 此外請設定這個環境變數

set QT_QPA_PLATFORM_PLUGIN_PATH=你的路徑\Sonicstride\WEBCAMGUI\Lib\site-packages\PyQt5\Qt5\plugins\platforms

set QT_QPA_PLATFORM_PLUGIN_PATH=C:\Users\NCKUI\OneDrive\桌面\Sonic-stride\Sonicstride\WEBCAMGUI\Lib\site-packages\PyQt5\Qt5\plugins\platforms
----


3. 在WEBCAMGUI的環境下輸入

python Sonicstride_GUI.py

就可以開始圖形介面的操作