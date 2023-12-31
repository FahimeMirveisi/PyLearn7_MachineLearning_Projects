import arcade
from snake import Snake
from apple import Apple


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
SCREEN_TITLE = "Super Snake V1"


# Class Game
class Game(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, resizable=True)
        arcade.set_background_color(arcade.color.KHAKI)
        self.game_background = arcade.load_texture("assets/game_background.png")
        self.game_over_background = arcade.load_texture("assets/game_over_background1.png")
        self.game_over = False
        self.apple = Apple(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.snake = Snake(SCREEN_WIDTH, SCREEN_HEIGHT)

    def on_draw(self):
        arcade.start_render()

        if not self.game_over:
            arcade.draw_texture_rectangle(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, SCREEN_WIDTH,
                                          SCREEN_HEIGHT, self.game_background)
            self.snake.draw()
            self.apple.draw()

            arcade.draw_text(f"SCORE : {self.snake.score}", SCREEN_WIDTH - 300,
                             SCREEN_HEIGHT - 30, arcade.color.YELLOW_ROSE, font_size= 20)

        elif self.game_over:
            arcade.draw_texture_rectangle(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, SCREEN_WIDTH,
                                          SCREEN_HEIGHT, self.game_over_background)
        arcade.finish_render()

    def on_update(self, delta_time: float):
        self.snake.move()
        #self.apple.on_update()

        if arcade.check_for_collision(self.snake, self.apple):
            self.snake.eat(self.apple)
            self.apple = Apple(SCREEN_WIDTH, SCREEN_HEIGHT)

        for part in self.snake.body:
            if self.snake.center_x == part["center_x"] and self.snake.center_y == part["center_y"]:
                self.game_over = True
                self.on_draw()

        if self.snake.center_x == 8 or self.snake.center_x == SCREEN_WIDTH - 8 or self.snake.center_y == 8 or self.snake.center_y == SCREEN_HEIGHT - 8:
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
