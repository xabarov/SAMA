from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PyQt5.QtWebEngineWidgets import QWebEngineSettings, QWebEngineView


class PdfViewer(QWidget):
    def __init__(self):
        super(PdfViewer, self).__init__()

        self.setWindowTitle("PDF Viewer")
        self.setGeometry(0, 28, 1000, 750)

        self.webView = QWebEngineView()
        self.webView.settings().setAttribute(QWebEngineSettings.PluginsEnabled, True)
        self.webView.settings().setAttribute(QWebEngineSettings.PdfViewerEnabled, True)

        self.mainLayout = QVBoxLayout()
        btnLayout = QVBoxLayout()

        self.okBtn = QPushButton('Ок', self)
        self.okBtn.clicked.connect(self.close)

        btnLayout.addWidget(self.okBtn)

        self.mainLayout.addWidget(self.webView)
        self.mainLayout.addLayout(btnLayout)
        self.setLayout(self.mainLayout)

    def url_changed(self):
        self.setWindowTitle(self.webView.title())

    def go_back(self):
        self.webView.back()

    def setPdf(self, url):
        self.webView.setUrl(QUrl(f"file://{url}"))
