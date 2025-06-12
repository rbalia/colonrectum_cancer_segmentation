class IndexTracker:
    def __init__(self, ax, X, slice_ax):
        self.ax = ax
        self.slice_ax = slice_ax
        #ax.set_title('use scroll wheel to navigate images')

        self.X = X
        shape = X.shape
        self.slices = shape[slice_ax]
        self.ind = self.slices // 2
        self.im = ax.imshow(self.get_slice(), vmin=0, vmax=1)#[:, :, self.ind])
        self.update()

    def on_scroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def get_slice(self):
        return self.X.take(self.ind, self.slice_ax)

    def update(self):
        self.im.set_data(self.get_slice())
        self.ax.set_xlabel('slice %s' % self.ind)
        #self.im.axes.figure.canvas.draw()
        self.im.axes.figure.canvas.draw_idle()
        #self.im.axes.figure.canvas.blit(self.ax.bbox)
