from pioneer.common.colors import amplitudes_to_color
from pioneer.das.api import platform

from yav import viewer

from sklearn.neighbors import KDTree

plat = platform.Platform('/nas/exportedDataset/20190319_162349_rec_dataset_townhall_to_top_o_frites_exported')
synched = plat.synchronized(['flir_tfc_img', 'eagle_tfc_ech'])
n = len(synched)

v = viewer(num=n)
eagle_tfc_wnd = v.create_point_cloud_window(title='eagle_tfc_ech')
eagle_pc = eagle_tfc_wnd.create_point_cloud()
eagle_sphere = eagle_tfc_wnd.create_sphere([0, 0, 0], radius=0.1, color=[128, 0, 0])
eagle_sphere.hide()

flir_tfc_wnd = v.create_image_window(title='flir_tfc_img')

pts = None
image_pts = None
kdtree = None

def on_update(v):
    global pts, image_pts, kdtree

    frame = v.get_frame()
    data = synched[frame]

    leddar_data = data['eagle_tfc_ech']
    pts = leddar_data.point_cloud()
    colors = amplitudes_to_color(leddar_data.amplitudes, log_normalize=True)

    eagle_pc.set_points(pts, colors)

    camera_pts = leddar_data.point_cloud(referential='flir_tfc')


    camera_data = data['flir_tfc_img']
    image_pts = camera_data.project_pts(camera_pts)
    if eagle_sphere.is_visibile():
        kdtree = KDTree(image_pts)

    image = camera_data.raw

    del flir_tfc_wnd.ax.collections[:]
    flir_tfc_wnd.update(image[..., ::-1])
    colors_f = colors.astype('f4')/255
    flir_tfc_wnd.ax.scatter(image_pts[:, 0], image_pts[:, 1], s=2, c=colors_f)
    flir_tfc_wnd.draw()

def on_key(v):
    global image_pts, kdtree
    visible = eagle_sphere.is_visibile()
    eagle_sphere.show(not visible)
    if eagle_sphere.is_visibile():
        kdtree = KDTree(image_pts)

def on_mouse_move(event):
    global pts, kdtree
    if eagle_sphere.is_visibile():
        if event.xdata is None or event.ydata is None:
            return
        query_pt = [[event.xdata, event.ydata]]
        # closest point
        ind = kdtree.query(query_pt, k=1, return_distance=False).ravel()
        eagle_sphere.set_center(pts[ind, :])
        eagle_tfc_wnd.render()

flir_tfc_wnd.canvas.mpl_connect('motion_notify_event', on_mouse_move)
v.add_frame_callback(on_update)
v.add_key_callback(on_key, 's')
v.run()
