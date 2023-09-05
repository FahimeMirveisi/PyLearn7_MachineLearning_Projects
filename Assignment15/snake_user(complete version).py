import random
import arcade


# Class Apple
class Apple(arcade.Sprite):
    def __init__(self, game):
        super().__init__("game images/apple.png")
        self.width = 30
        self.height = 30
        self.center_x = random.randint(50, game.width - 50)
        self.center_y = random.randint(50, game.height - 50)
        self.change_x = 0
        self.change_y = 0


# Class Pear
class Pear(arcade.Sprite):
    def __init__(self, game):
        super().__init__("game images/pear.png")
        self.width = 30
        self.height = 30
        self.center_x = random.randint(50, game.width - 50)
        self.center_y = random.randint(50, game.height - 50)
        self.change_x = 0
        self.change_y = 0


# Class Poop
class Poop(arcade.Sprite):
    def __init__(self, game):
        super().__init__("game images/poop_glasses.png")
        self.width = 30
        self.height = 30
        self.center_x = random.randint(50, game.width - 50)
        self.center_y = random.randint(50, game.height - 50)
        self.change_x = 0
        self.change_y = 0


# Class Snake
class Snake(arcade.Sprite):
    def __init__(self, game):
        super().__init__()
        self.width = 40
        self.height = 40
        self.center_x = game.width // 2
        self.center_y = game.height // 2
        self.head = arcade.load_texture("game images/snake head.png")
        self.color1 = arcade.color.GREEN
        self.color2 = arcade.color.BLACK
        self.change_x = 0
        self.change_y = 0
        self.speed = 4
        self.score = 0
        self.body = []


    def draw(self):
        # head

        arcade.draw_texture_rectangle(self.center_x, self.center_y, self.width, self.height, self.head)
        body_lenght = 0
        # body
        for part in self.body:
            if body_lenght % 2 == 0:
                arcade.draw_rectangle_outline(part['x'], part['y'], self.width, self.height , self.color1)

            else:
                arcade.draw_rectangle_outline(part['x'], part['y'], self.width, self.height, self.color2)

            body_lenght += 1

    def move(self):
        self.body.append({'x': self.center_x , 'y': self.center_y })
        if len(self.body) > self.score:
            self.body.pop(0)
        self.center_x += self.change_x * self.speed
        self.center_y += self.change_y * self.speed

    def eat(self, game):
        if game.food == "apple":
            del game.apple
            self.score += 1
        elif game.food == "pear":
            del game.pear
            self.score += 2
        elif game.food == "poop":
            del game.poop
            self.score -= 1

        print("Score:", self.score)


# Class Game
class Game(arcade.Window):
    def __init__(self):
        super().__init__(width=500, height=500, title="Super Snake V1")
        arcade.set_background_color(arcade.color.KHAKI)
        self.game_background = arcade.load_texture("game images/game_background.png")
        self.game_over_background = arcade.load_texture("game images/game_over_background1.png")
        self.game_over = False
        self.food = "apple"
        self.apple = Apple(self)
        self.pear = Pear(self)
        self.poop = Poop(self)
        self.snake = Snake(self)

    def on_draw(self):
        arcade.start_render()
        if not self.game_over:
            arcade.draw_texture_rectangle(self.width // 2, self.height // 2, self.width,
                                          self.height, self.game_background)
            self.snake.draw()
            self.apple.draw()
            self.pear.draw()
            self.poop.draw()

            arcade.draw_text(f"SCORE : {self.snake.score}", self.width - 300,
                             self.height - 30, arcade.color.YELLOW_ROSE, font_size= 20)

        elif self.game_over:
            arcade.draw_texture_rectangle(self.width // 2, self.height // 2, self.width,
                                          self.height, self.game_over_background)
        arcade.finish_render()

    def on_update(self, delta_time: float):
        self.snake.move()

        if arcade.check_for_collision(self.snake, self.apple):
            self.food = "apple"
            self.snake.eat(self)
            self.apple = Apple(self)

        if arcade.check_for_collision(self.snake, self.pear):
            self.food = "pear"
            self.snake.eat(self)
            self.pear = Pear(self)

        if arcade.check_for_collision(self.snake, self.poop):
            self.food = "poop"
            self.snake.eat(self)
            self.poop = Poop(self)

        for part in self.snake.body:
            if self.snake.center_x == part["x"] and self.snake.center_y == part["y"]:
                self.game_over = True
                self.on_draw()

        if self.snake.center_x == 10 or self.snake.center_x == self.width - 10 or self.snake.center_y == 10 or\
                self.snake.center_y == self.height - 10:
            self.game_over = True
            self.on_draw()

        if self.snake.score == -1:
            self.game_over = True
            self.on_draw()

    def on_key_release(self, symbol: int, modifiers: int):
        if symbol == arcade.key.UP:
            self.snake.change_x = 0
            self.snake.change_y = 1
        elif symbol == arcade.key.DOWN:
            self.snake.change_x = 0
            self.snake.change_y = -1
        elif symbol == arcade.key.RIGHT:
            self.snake.change_x = 1
            self.snake.change_y = 0
        elif symbol == arcade.key.LEFT:
            self.snake.change_x = -1
            self.snake.change_y = 0


if __name__ == "__main__":
    game = Game()
    arcade.run()