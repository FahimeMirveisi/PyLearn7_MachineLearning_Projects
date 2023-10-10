import random
import arcade

class Spaceshipe(arcade.Sprite):
    def __init__(self, game):
        super().__init__(':resources:images/space_shooter/playerShip1_orange.png')
        self.center_x = game.width // 2
        self.center_y = game.height // 2
        self.width = 64
        self.width = 64
        self.speed = 8


class Enemy(arcade.Sprite):
    def __init__(self, game):
        super().__init__(':resources:images/space_shooter/playerShip1_green.png')
        self.center_x = random.randint(0, game.width)
        self.center_y = game.height + 40
        self.angle = 180
        self.speed = 4


class Game(arcade.Window):
    def __init__(self):
        super().__init__(width=800, height=600, title='Interstellar Game')
        arcade.set_background_color(arcade.color.BLACK)
        self.background = arcade.load_texture(":resources:images/backgrounds/stars.png")
        self.spaceship = Spaceshipe(self)
        self.enemy = Enemy(self)

    def on_draw(self):
        arcade.start_render()
        arcade.draw_lrwh_rectangle_textured(0, 0, self.width, self.height, self.background)
        self.spaceship.draw()
        self.enemy.draw()

    def on_key_press(self, symbol: int, modifiers: int):
        # print(symbol)
        if symbol == 97:
            self.spaceship.center_x = self.spaceship.center_x - self.spaceship.speed
        elif symbol == 100:
            self.spaceship.center_x = self.spaceship.center_x + self.spaceship.speed

    def on_update(self, delta_time: float):
        self.enemy.center_y -= self.enemy.speed

window = Game()
arcade.run()