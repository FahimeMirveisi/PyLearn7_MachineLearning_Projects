import arcade
from game_objects import Apple
from game_objects import Pear
from game_objects import Shit
from snake import Snake


class Game(arcade.Window):
    def __init__(self):
        super().__init__(width=600, height=600, title="Snake game 🐍 (Version 1)")
        arcade.set_background_color(arcade.color.KHAKI)
        self.game_background = arcade.load_texture("game images/game_background.png")
        self.game_over_background = arcade.load_texture("game images/game_over_background1.png")
        self.status = "is playing"
        self.food = Apple(self)


    def on_draw(self):
        arcade.start_render()

        if self.status == "is playing":
            arcade.draw_texture_rectangle(self.width // 2, self.height // 2, self.width, self.height,
                                          self.game_background)
            self.food.draw()
        elif self.status == "game over":
            arcade.draw_texture_rectangle(self.width // 2, self.height // 2, self.width, self.height,
                                          self.game_over_background)


        arcade.finish_render()


if __name__ == "__main__":
    game = Game()
    arcade.run()