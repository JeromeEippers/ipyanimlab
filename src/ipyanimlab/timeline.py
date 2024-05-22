from ipywidgets import widgets 
from traitlets import (
    Unicode, CInt, Bool, CaselessStrEnum, Tuple, TraitError, default, validate
)


class Timeline (widgets.ValueWidget, widgets.HBox):
    """A timeline play/pause/stop with slider for Jupiter"""

    value = CInt(0, help="Int value").tag(sync=True)

    def __init__(self, **kwargs):
        kwargs['value'] = 0
        self._play = widgets.Play(
                step=1,
                interval=1000.0/30.0,
                description="play",
                disabled=False,
                layout=widgets.Layout(padding='0px 0px 0px 0px'),
                **kwargs
        )

        self._slider = widgets.IntSlider(layout=widgets.Layout(width='850px', padding='0px 0px 0px 0px'), **kwargs)
        super().__init__(children=[self._play, self._slider], **kwargs)
        widgets.jslink((self._play, 'value'), (self._slider, 'value'))

        self._slider.observe(self.my_value, names='value')

    def my_value(self, change):
        self.value = self._slider.value

    def set_value(self, value):
        self._slider.value = value

    @property
    def max(self):
        return self._slider.max

    @max.setter
    def max(self, value):
        self._play.max = value
        self._slider.max = value
        
    @property
    def min(self):
        return self._slider.min

    @min.setter
    def min(self, value):
        self._play.min = value
        self._slider.min = value
