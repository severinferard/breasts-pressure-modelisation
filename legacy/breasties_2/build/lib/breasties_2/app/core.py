
from trame.app import get_server
from trame.decorators import TrameApp, change, controller
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3, vtk
from breasties_2.widgets import breasties_2 as my_widgets

import pyvista as pv
from pyvista import examples
from pyvista.trame.ui import plotter_ui
import numpy as np
from scipy.spatial import KDTree
import os
from threading import Timer
import asyncio
import time
# pv.OFF_SCREEN = True

# ---------------------------------------------------------
# Engine class
# ---------------------------------------------------------

MAX_VALUE = 150
MIN_VALUE = 30

N_COLUMNS = 21
N_ROWS = 12
K = 5


@TrameApp()
class MyTrameApp:
    def __init__(self, server=None):
        self.server = get_server(server, client_type="vue3")
        if self.server.hot_reload:
            self.server.controller.on_server_reload.add(self._build_ui)
        self.ui = self._build_ui()

        # Set state variable
        self.state.trame__title = "Breasties2.0"
        self.state.resolution = 6

        while True:
            print("TEST")
            time.sleep(1)

    @property
    def state(self):
        return self.server.state

    @property
    def ctrl(self):
        return self.server.controller

    @controller.set("reset_resolution")
    def reset_resolution(self):
        self.state.resolution = 6

    @change("resolution")
    def on_resolution_change(self, resolution, **kwargs):
        print(f">>> ENGINE(a): Slider updating resolution to {resolution}")

    @controller.set("widget_click")
    def widget_click(self):
        print(">>> ENGINE(a): Widget Click")

    @controller.set("widget_change")
    def widget_change(self):
        print(">>> ENGINE(a): Widget Change")

    def callback(self):
        self.boob_mesh += 1
        print("HELLO")

    # @asynchronous.task
    async def start_countdown(self):
        while True:
            print("ksjfbsdbf")
        # try:
        #     state.countdown = int(state.countdown)
        # except:
        #     state.countdown = countdown_init

        # while state.keep_updating:
        #     with state:
        #         await asyncio.sleep(10.0)
        #         print("keep updating = ", state.keep_updating)
        #         global history_filename
        #         readHistory(history_filename)

        #         # state.countdown = not state.countdown
        #         state.countdown -= 1

    def _build_ui(self, *args, **kwargs):

        mesh = examples.load_random_hills()

        # print("self.pl", self.pl)
        # pl = pv.Plotter()
        # self.pl.add_mesh(mesh)

        dir = os.path.dirname(__file__)
        assets_dir = os.path.join(dir, '../../../../../../assets/')
        # print("assets_dir", assets_dir)
        points = np.load(os.path.join(assets_dir, 'boob_grid_12_21.npy'))

        # Remove points that are outside of the boob mesh.
        # Because those points have been skipped during the points mapping, their value is (0, 0, 0)
        origin = np.array([0, 0, 0])
        points_to_skip = np.where((points == origin).all(axis=1))[0]
        points_of_interest = np.delete(points, points_to_skip, axis=0)

        # Create the point cloud and initialize the scalars
        point_cloud = pv.PolyData(points_of_interest)
        point_cloud['pressure'] = np.zeros(points_of_interest.shape[0])
        point_cloud['max_pressure'] = np.zeros(points_of_interest.shape[0])

        # Load the boob 3D model from file
        self.boob_mesh = pv.read(os.path.join(assets_dir, 'boob.obj'))
        self.boob_mesh.translate(np.array([0, -0.035, 0]), inplace=True)

        # Create a kd-tree for quick nearest-neighbor lookup.
        kdtree = KDTree(points_of_interest)

        # Find the K nearest point_cloud points for each points in the boob mesh and calculate their respective distances
        self.boob_mesh['nearest_points'] = kdtree.query(self.boob_mesh.points, k=K)[1]
        self.boob_mesh['nearest_points_dist'] = kdtree.query(self.boob_mesh.points, k=K)[0]

        # Step 1: Calculate the reciprocal of the distances
        reciprocal_distances = 1 / self.boob_mesh['nearest_points_dist']

        # Step 2: Square the reciprocal distances
        squared_reciprocal_distances = reciprocal_distances ** 1.2

        # Step 3: Normalize the weights
        normalized_weights = squared_reciprocal_distances / \
            np.sum(squared_reciprocal_distances, axis=1, keepdims=True)

        self.boob_mesh['pressure'] = np.zeros(self.boob_mesh.points.shape[0])
        self.boob_mesh['max_pressure'] = self.boob_mesh['pressure']

        self.pl = pv.Plotter(shape=(1, 2), border=False)
        self.pl.subplot(0, 0)
        self.pl.add_text("Measured Pressure Values on Grid", font_size=12)
        self.pl.add_mesh(point_cloud, scalars='pressure', cmap='cool',
                         point_size=15, clim=[MIN_VALUE, MAX_VALUE])

        self.pl.subplot(0, 1)
        self.pl.add_text(f"Inverse Distance Weighted Interpolation (k={K})", font_size=12)
        self.pl.add_mesh(self.boob_mesh, scalars='pressure', cmap='cool', clim=[MIN_VALUE, MAX_VALUE])

        # pass

        # self.pl.iren.add_timer_event(max_steps=1000, duration=50, callback=self.callback)
        # self.pl.iren.interactor.CreateRepeatingTimer(50)
        # self.pl.iren.interactor.AddObserver("TimerEvent", lambda *_: print('TimerEvent'))
        # self.pl.iren.add_observer(
        #     "TimerEvent", lambda *_: print('TimerEvent')
        # )

        # self.pl.iren.create_timer(1000);
        # self.pl.iren

        print(">>> BUILD UI")

        # self.pl.iren.interactor.AddObserver('TimerEvent', self.callback, 10)

        # pl.iren.create_timer(duration=50)

        # pl.add_timer_event(duration=50, callback=callback, max_steps=np.iinfo(np.int32).max)
        self.pl.link_views()

        # timer = Timer(100, self.callback)
        # timer.start()

        with SinglePageLayout(self.server) as layout:
            # Toolbar
            layout.title.set_text("Breasties 2.0")
            # with layout.toolbar:
            #     vuetify3.VSpacer()
            #     my_widgets.CustomWidget(
            #         attribute_name="Hello",
            #         py_attr_name="World",
            #         click=self.ctrl.widget_click,
            #         change=self.ctrl.widget_change,
            #     )
            #     vuetify3.VSpacer()
            #     vuetify3.VSlider(                    # Add slider
            #         v_model=("resolution", 6),      # bind variable with an initial value of 6
            #         min=3, max=60, step=1,          # slider range
            #         dense=True, hide_details=True,  # presentation setup
            #     )
            #     with vuetify3.VBtn(icon=True, click=self.ctrl.reset_camera):
            #         vuetify3.VIcon("mdi-crop-free")
            #     with vuetify3.VBtn(icon=True, click=self.reset_resolution):
            #         vuetify3.VIcon("mdi-undo")

            # Main content
            with layout.content:
                with vuetify3.VContainer(fluid=True, classes="pa-0 fill-height"):
                    view = plotter_ui(self.pl, add_menu=False, mode='server')
                    # self.pl.iren.interactor.CreateRepeatingTimer(50)
                    # self.pl.iren.interactor.AddObserver("TimerEvent", lambda *_: print('TimerEvent'))
                    # with vtk.VtkView() as vtk_view:                # vtk.js view for local rendering
                    #     self.ctrl.reset_camera = vtk_view.reset_camera  # Bind method to controller
                    #     with vtk.VtkGeometryRepresentation():      # Add representation to vtk.js view
                    #         vtk.VtkAlgorithm(                      # Add ConeSource to representation
                    #             vtk_class="vtkConeSource",          # Set attribute value with no JS eval
                    #             state=("{ resolution }",)          # Set attribute value with JS eval
                    #         )

            # Footer
            # layout.footer.hide()

            return layout
