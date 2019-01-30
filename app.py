from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty
from kivy.vector import Vector
from linear_regression import GenerateLR
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
from kivy.uix.label import Label



class MyButton(ButtonBehavior, Label):
    def __init__(self, **kwargs):
        super(MyButton, self).__init__(**kwargs)



    def on_press(self):
        self.source = 'test.png'

    def on_release(self):
        self.source = 'test.png'



class LinnearRegressionApp(App):
    def build(self):
        return MyButton()


if __name__ == '__main__':
    LinnearRegressionApp().run()
