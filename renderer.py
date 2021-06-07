from camera import *
from projection import *
from object import *

class SoftwareRenderer:
    objects: Object = []

    def __init__(self):
        pg.init()
        self.RES = self.WIDTH, self.HEIGHT = 900, 600
        self.FPS = 60
        self.H_WIDTH, self.H_HEIGHT = self.WIDTH // 2, self.HEIGHT // 2
        self.screen = pg.display.set_mode(self.RES)
        self.clock = pg.time.Clock()
        
    def bind_camera(self, x, y, z):
        self.camera = Camera(self, [x, y, z])
        self.projection = Projection(self)

    def init_object(self, obj):
        self.objects.append(obj)

    def delete_object(self, idx=0):
        self.objects.pop()

    def draw(self):
        self.screen.fill(pg.Color('White'))
        for obj in self.objects:
            obj.draw()
        self.camera.control()

    # def run(self):
    #     while True:
    #         self.draw()
    #         [exit() for i in pg.event.get() if i.type == pg.constants.QUIT]
    #         pg.display.set_caption(str(self.clock.get_fps()))
    #         # manager.redraw()
    #         pg.display.flip()
    #         self.clock.tick(self.FPS)