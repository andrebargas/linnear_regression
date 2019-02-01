from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.label import Label
from linnear_regression import generateLRvolume, generateLRfechamento


# Create both screens. Please note the root.manager.current: this is how
# you can control the ScreenManager from kv. Each screen has by default a
# property manager that gives you the instance of the ScreenManager used.
Builder.load_string("""
<MenuScreen>:
    BoxLayout:
        orientation:'vertical'

        Button:
            text: 'Informações do projeto'
            on_press: root.manager.current = 'about'

        Button:
            text: 'Gerar grafico de para prever fechamento'
            on_press: root.generateFechamentoMPL()
        Button:
            text: 'Gerar grafico de para prever volume'
            on_press: root.generateVolumeMPL()

        Button:
            text: 'Quit'
            on_press: app.stop()
<AboutScreen>:
    BoxLayout:
        orientation:'vertical'

        Label:
            text: "Sobre o Projeto :"
            font_size: 30
        Label:
            text: "O objetivo deste projeto fora o aprendizado a respeito de técnicas de inteligência artificial, aplicadas a datasets de preço de ações em uma bolsa do mercado. Para isso foi utilizado o web service do Quandl disponivel em (https://www.quandl.com/), além das bibliotecas da sklearn, pandas numpy e matplotlib para visualização dos resultados."
            text_size: cm(15), cm(4)
            collor: 'white'
        Button:
            text: 'Voltar'
            on_press: root.manager.current = 'menu'
""")

# Declare both screens
class MenuScreen(Screen):
    def generateFechamentoMPL(self):
        generateLRfechamento()
    def generateVolumeMPL(self):
        generateLRvolume()

    pass

class AboutScreen(Screen):
    # def __init__(self, **kwargs):
    #     self.add_widget(Label(text='Hello world'))
    pass



# Create the screen manager
sm = ScreenManager()
sm.add_widget(MenuScreen(name='menu'))
sm.add_widget(AboutScreen(name='about'))

class TestApp(App):

    def build(self):
        return sm

if __name__ == '__main__':
    TestApp().run()
