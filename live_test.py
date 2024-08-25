from __future__ import annotations

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

from serial.tools import list_ports

import pyvista as pv
from pyvista import examples
import pyvistaqt as pvqt
from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import QComboBox, QMainWindow, QApplication, QWidget, QVBoxLayout, QWidgetAction
from PyQt5.QtGui import QIcon


np.set_printoptions(linewidth=500)

N_COLUMNS = 12
N_ROWS = 21

st_helens = examples.download_st_helens().warp_by_scalar()


def onPress():
    pass


# def getAvailableSerialPorts():
    # print(serial.tools.list_ports())

class Plotter(pvqt.BackgroundPlotter):
    def __init__(self):
        super().__init__()

        self.serialPortsCombo = QComboBox()

        serialPortsAction = QWidgetAction(self.app_window)
        serialPortsAction.setDefaultWidget(self.serialPortsCombo)
        self.userToolbar = self.app_window.addToolBar('User Toolbar')
        self.userToolbar.addAction(serialPortsAction)

        self.updateSerialPorts()

    def updateSerialPorts(self):
        for i in range(self.serialPortsCombo.count()):
            self.serialPortsCombo.removeItem(i)

        for port in list_ports.comports():
            self.serialPortsCombo.addItem(port.name)


def main():
    pl = Plotter()

    # combobox1 = QComboBox()
    # for port in list_ports.comports():
    #     combobox1.addItem(port.name)

    # combobox1.removeItem()
    # user_toolbar = pl.app_window.addToolBar('User Toolbar')
    # action = QWidgetAction(pl.app_window)
    # action.setDefaultWidget(combobox1)
    # user_toolbar.addAction(action)

    pl.add_mesh(st_helens)

    # while True:
    #     pl.render()
    #     pl.app.processEvents()

    while True:
        pl.render()
        pl.app.processEvents()


if __name__ == '__main__':
    main()
