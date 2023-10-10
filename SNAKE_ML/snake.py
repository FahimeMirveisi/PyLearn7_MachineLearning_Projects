
import arcade


# class snake
class Snake(arcade.Sprite):
    def __init__(self, SCREEN_WIDTH, SCREEN_HEIGHT):
        super().__init__()

        self.width = 16
        self.height = 16
        self.center_x = SCREEN_WIDTH // 2
        self.center_y = SCREEN_HEIGHT // 2
        self.head = arcade.load_texture("assets/snake head.png")
        self.color1 = arcade.color.GREEN
        self.color2 = arcade.color.BLACK
        self.change_x = 1
        self.change_y = 1
        self.speed = 8
        self.score = 0
        self.body = []

    def draw(self):

        # head
        arcade.draw_texture_rectangle(self.center_x, self.center_y, self.width, self.height, self.head)

        # body
        for body_length, part in enumerate(self.body):
            if body_length % 2 == 0:
                arcade.draw_rectangle_filled(part['center_x'], part['center_y'], self.width, self.height, self.color1)

            else:
                arcade.draw_rectangle_filled(part['center_x'], part['center_y'], self.width, self.height, self.color2)

    def move(self):
        self.body.append({'center_x': self.center_x, 'center_y': self.center_y})
        if len(self.body) > self.score:
            self.body.pop(0)
        self.center_x += self.change_x * self.speed
        self.center_y += self.change_y * self.speed

    def on_update(self, delta_time: float = 1 / 60):
        self.body.append({'center_x': self.center_x, 'center_y': self.center_y})
        if len(self.body) > self.score:
            self.body.pop(0)

        self.center_x += self.change_x * self.speed
        self.center_y += self.change_y * self.speed

    def eat(self, apple):
        del apple
        self.score += 1
        print("Score:", self.score)
