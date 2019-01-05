#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import wx
import subprocess as sp
import numpy as np


class FilterPass:
    def __init__(self, ref, hue=None, rgb=None, lum_min=None, lum_max=None,
                 sat_min=None, sat_max=None):
        self.ref = ref
        self.hue = hue
        self.rgb = rgb
        self.lum_min = lum_min
        self.lum_max = lum_max
        self.sat_min = sat_min
        self.sat_max = sat_max

    def __call__(self, img):
        img = img.astype(np.int16)
        ret = np.ones(img.shape[:2], np.bool_)
        if self.rgb is not None:
            ret = np.logical_and(
                ret, np.abs(img - self.ref).sum(2) <= self.rgb * 3)
        if (self.hue is not None
                or self.lum_min is not None or self.lum_max is not None
                or self.sat_min is not None or self.sat_max is not None):
            img = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_RGB2HSV)
        if self.hue is not None:
            hue_diff = (img[:, :, 0] -
                        cv2.cvtColor(np.array([[self.rgb]], np.float32) / 255,
                                     cv2.COLOR_RGB2HSV)[0, 0, 0]) % 360
            ret = np.logical_and(ret, np.logical_or(hue_diff <= self.hue,
                                                    hue_diff >= 360 - self.hue))
        if self.sat_min is not None:
            ret = np.logical_and(ret, img[:, :, 1] >= self.sat_min / 100)
        if self.sat_max is not None:
            ret = np.logical_and(ret, img[:, :, 1] <= self.sat_max / 100)
        if self.lum_min is not None:
            ret = np.logical_and(ret, img[:, :, 2] >= self.lum_min / 100)
        if self.lum_max is not None:
            ret = np.logical_and(ret, img[:, :, 2] <= self.lum_max / 100)
        return ret


class Filter:
    def __init__(self, debug, outline, pass1, final):
        self.debug = debug
        self.outline = outline
        self.pass1 = pass1
        self.final = final

    def __call__(self, img):
        ret = np.full(img.shape, 255)
        if self.outline:
            pass1 = np.logical_not(self.pass1(img))
            outline = self.outline(img)
            ret[outline] = [255, 0, 0]
            for i, j in zip(*pass1.nonzero()):
                if ret[i, j, 2]:
                    cv2.floodFill(ret, None, (j, i), (0, 128, 0))
            ret[pass1] = 0

        sub = (ret == 255).all(2)
        final = self.final(img)
        ret[np.logical_and(sub, np.logical_not(final))] = [0, 0, 255]

        if not self.debug:
            ret = (ret == 255).all(2, keepdims=True).astype(np.uint8) * 255

        return ret


class ColorButton(wx.Button):
    def __init__(self, parent, log, size, disable=False):
        wx.Button.__init__(self, parent, log, size=(size, size))
        self.SetBackgroundColour((255, 255, 255))
        if disable:
            self.Disable()
        else:
            self.Bind(wx.EVT_BUTTON, self.on_button)
        self.parent = parent

    def on_button(self, event):
        dlg = wx.ColourDialog(self.parent)
        dlg.GetColourData().SetChooseFull(True)
        if dlg.ShowModal() == wx.ID_OK:
            data = dlg.GetColourData()
            color = data.GetColour().Get()
            self.SetBackgroundColour(color)


class FilterBox(wx.StaticBoxSizer):
    def __init__(self, parent, label):
        wx.StaticBoxSizer.__init__(self, wx.VERTICAL, parent, label)
        self.hue_b = wx.CheckBox(parent, -1, 'Hue Difference')
        self.hue_v = wx.SpinCtrl(parent, -1, max=180)
        spin_size = self.hue_v.GetSizeFromTextSize(
            self.hue_v.GetTextExtent('000'))
        self.hue_v.SetMinSize(spin_size)
        self.hue_v.SetSize(spin_size)
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(self.hue_b, 1, wx.ALL, 3)
        row.Add(self.hue_v, 0, wx.ALL, 3)
        self.Add(row, 0, wx.EXPAND)
        self.rgb_b = wx.CheckBox(parent, -1, 'RGB Difference')
        self.rgb_v = wx.SpinCtrl(parent, -1, size=spin_size, max=255)
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(self.rgb_b, 1, wx.ALL, 3)
        row.Add(self.rgb_v, 0, wx.ALL, 3)
        self.Add(row, 0, wx.EXPAND)
        self.lum_min_b = wx.CheckBox(parent, -1, 'Lum Min')
        self.lum_min_v = wx.SpinCtrl(parent, -1, size=spin_size, max=100)
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(self.lum_min_b, 1, wx.ALL, 3)
        row.Add(self.lum_min_v, 0, wx.ALL, 3)
        self.Add(row, 0, wx.EXPAND)
        self.lum_max_b = wx.CheckBox(parent, -1, 'Lum Max')
        self.lum_max_v = wx.SpinCtrl(parent, -1, size=spin_size, max=100)
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(self.lum_max_b, 1, wx.ALL, 3)
        row.Add(self.lum_max_v, 0, wx.ALL, 3)
        self.Add(row, 0, wx.EXPAND)
        self.sat_min_b = wx.CheckBox(parent, -1, 'Sat Min')
        self.sat_min_v = wx.SpinCtrl(parent, -1, size=spin_size, max=100)
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(self.sat_min_b, 1, wx.ALL, 3)
        row.Add(self.sat_min_v, 0, wx.ALL, 3)
        self.Add(row, 0, wx.EXPAND)
        self.sat_max_b = wx.CheckBox(parent, -1, 'Sat Max')
        self.sat_max_v = wx.SpinCtrl(parent, -1, size=spin_size, max=100)
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(self.sat_max_b, 1, wx.ALL, 3)
        row.Add(self.sat_max_v, 0, wx.ALL, 3)
        self.Add(row, 0, wx.EXPAND)

    def get_filter(self):
        return FilterPass(
            None,
            self.hue_v.GetValue() if self.hue_b.GetValue() else None,
            self.rgb_v.GetValue() if self.rgb_b.GetValue() else None,
            self.lum_min_v.GetValue() if self.lum_min_b.GetValue() else None,
            self.lum_max_v.GetValue() if self.lum_max_b.GetValue() else None,
            self.sat_min_v.GetValue() if self.sat_min_b.GetValue() else None,
            self.sat_max_v.GetValue() if self.sat_max_b.GetValue() else None)


class FilterAdvanceFrame(wx.Frame):
    def __init__(self, parent, log, callback):
        self.callback = callback
        wx.Frame.__init__(self, parent, log, 'Filter Setting',
                          style=wx.SYSTEM_MENU | wx.CAPTION | wx.CLIP_CHILDREN)
        p = wx.Panel(self, -1)
        self.p = p
        filter_box = wx.BoxSizer(wx.HORIZONTAL)
        self.outline = FilterBox(p, 'Outline')
        self.pass1 = FilterBox(p, 'Pass1')
        self.final = FilterBox(p, 'Final')
        filter_box.Add(self.outline, 1, wx.ALL, 5)
        filter_box.Add(self.pass1, 1, wx.ALL, 5)
        filter_box.Add(self.final, 1, wx.ALL, 5)
        button = wx.Button(p, -1, 'OK')
        button.Bind(wx.EVT_BUTTON, self.on_button)
        box = wx.BoxSizer(wx.VERTICAL)
        box.Add(filter_box, 1, wx.EXPAND)
        box.Add(button, 0, wx.ALIGN_CENTER)
        p.Bind(wx.EVT_CHECKBOX, self.on_modify)
        p.Bind(wx.EVT_SPINCTRL, self.on_modify)
        p.SetSizer(box)
        box.Fit(p)
        self.Fit()

    def on_modify(self, event):
        self.callback()

    def on_button(self, event):
        self.Hide()


class PostprocessingFrame(wx.Frame):
    def __init__(self, parent, log, callback):
        self.callback = callback
        wx.Frame.__init__(self, parent, log, 'Postprocessing Setting',
                          style=wx.SYSTEM_MENU | wx.CAPTION | wx.CLIP_CHILDREN)
        p = wx.Panel(self, -1)
        self.p = p
        box = wx.BoxSizer(wx.VERTICAL)
        self.dot = wx.CheckBox(p, -1, 'Remove single pixel dot')
        self.dot.Disable()  # TODO
        box.Add(self.dot, 0, wx.EXPAND | wx.ALL, 5)
        self.line = wx.CheckBox(p, -1, 'Remove single pixel line')
        self.line.Disable()  # TODO
        box.Add(self.line, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        self.block = wx.CheckBox(p, -1, 'Remove block larger than')
        self.block.Disable()  # TODO
        box.Add(self.block, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        self.edge = wx.CheckBox(p, -1, 'Remove block touch edge')
        self.edge.Disable()  # TODO
        box.Add(self.edge, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        self.center = wx.CheckBox(p, -1, 'Remove block pass center')
        self.center.Disable()  # TODO
        box.Add(self.center, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        button = wx.Button(p, -1, 'OK')
        button.Bind(wx.EVT_BUTTON, self.on_button)
        box.Add(button, 0, wx.ALIGN_CENTER)
        p.SetSizer(box)
        box.Fit(p)
        self.Fit()

    def on_button(self, event):
        self.Hide()


class FilterFrame(wx.Frame):
    STATUS_NONE = 0
    STATUS_REGION = 1
    STATUS_FILTER = 2

    def __init__(self, parent, log, video, frame_count):
        self.video = video
        video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        wx.Frame.__init__(self, parent, log, "Filter")
        p = wx.Panel(self, -1)
        self.advance = FilterAdvanceFrame(p, -1, self.update_filter)
        self.postprocessing = PostprocessingFrame(p, -1, self.update_filter)
        self.filter = None
        self.p = p
        box_left = wx.BoxSizer(wx.VERTICAL)
        self.zoom_box = wx.StaticBoxSizer(wx.VERTICAL, p, 'Zoom')
        self.zoom_spin = wx.SpinCtrlDouble(p, -1, min=10, max=1000,
                                           initial=100,
                                           inc=10)
        self.zoom_spin.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_spinctrl)
        self.zoom_box.Add(self.zoom_spin, 0, wx.EXPAND | wx.ALL, 5)
        region_box = wx.StaticBoxSizer(wx.VERTICAL, p, 'Region')
        self.top_spin = wx.SpinCtrl(p, -1, max=video_height - 11, initial=0)
        spin_size = self.top_spin.GetSizeFromTextSize(
            self.top_spin.GetTextExtent("0000"))
        self.top_spin.SetMinSize(spin_size)
        self.top_spin.SetSize(spin_size)
        self.top_spin.Bind(wx.EVT_SPINCTRL, self.on_region_spinctrl)
        region_top_box = wx.BoxSizer(wx.HORIZONTAL)
        region_top_box.Add(wx.StaticText(p, -1), 1)
        region_top_box.Add(self.top_spin, 2)
        region_top_box.Add(wx.StaticText(p, -1), 1)
        region_box.Add(region_top_box, 0, wx.EXPAND | wx.ALL, 2)
        region_width_box = wx.BoxSizer(wx.HORIZONTAL)
        self.left_spin = wx.SpinCtrl(p, -1, size=spin_size,
                                     style=wx.SP_HORIZONTAL,
                                     max=video_width - 11, initial=0)
        self.left_spin.Bind(wx.EVT_SPINCTRL, self.on_region_spinctrl)
        region_width_box.Add(self.left_spin, 1, wx.ALL, 3)
        self.right_spin = wx.SpinCtrl(p, -1, size=spin_size,
                                      style=wx.SP_HORIZONTAL, min=10,
                                      max=video_width - 1,
                                      initial=video_width - 1)
        self.right_spin.Bind(wx.EVT_SPINCTRL, self.on_region_spinctrl)
        region_width_box.Add(self.right_spin, 1, wx.ALL, 3)
        region_box.Add(region_width_box, 0, wx.EXPAND)
        region_bottom_box = wx.BoxSizer(wx.HORIZONTAL)
        region_bottom_box.Add(wx.StaticText(p, -1), 1)
        self.bottom_spin = wx.SpinCtrl(p, -1, size=spin_size,
                                       max=video_height - 1,
                                       initial=video_height - 1)
        self.bottom_spin.Bind(wx.EVT_SPINCTRL, self.on_region_spinctrl)
        region_bottom_box.Add(self.bottom_spin, 2)
        region_bottom_box.Add(wx.StaticText(p, -1), 1)
        region_box.Add(region_bottom_box, 0, wx.EXPAND | wx.ALL, 2)
        full_width_button = wx.CheckBox(p, -1, 'Full Width')
        full_width_button.Disable()  # TODO
        region_box.Add(full_width_button, 0, wx.EXPAND | wx.ALL, 5)
        filter_settings_box = wx.StaticBoxSizer(wx.VERTICAL, p,
                                                'Filter Settings')
        self.filter_checkbox = wx.CheckBox(p, -1, 'Enable Filter')
        self.filter_checkbox.Bind(wx.EVT_CHECKBOX, self.on_filter_checkbox)
        filter_settings_box.Add(self.filter_checkbox, 0, wx.EXPAND | wx.ALL,
                                5)
        self.acolor_checkbox = wx.CheckBox(p, -1, 'Additional Color')
        self.acolor_checkbox.Bind(wx.EVT_CHECKBOX, self.on_filter_detail)
        filter_settings_box.Add(self.acolor_checkbox, 0,
                                wx.EXPAND | wx.ALL, 5)
        self.style_choice = wx.Choice(p, -1,
                                      choices=('Color', 'Color + Outline'))
        self.style_choice.Bind(wx.EVT_CHOICE, self.on_filter_detail)
        self.style_choice.SetSelection(0)
        filter_settings_box.Add(self.style_choice, 0, wx.EXPAND | wx.ALL, 5)
        filter_subtitle_box = wx.BoxSizer(wx.HORIZONTAL)
        subtitle_color_static = wx.StaticText(p, -1, 'Subtitle Color')
        colorbox_size = subtitle_color_static.GetSize()[1]
        filter_subtitle_box.Add(subtitle_color_static, 1)
        self.subtitle_color = ColorButton(p, -1, colorbox_size)
        filter_subtitle_box.Add(self.subtitle_color, 0)
        filter_settings_box.Add(filter_subtitle_box, 0, wx.EXPAND | wx.ALL,
                                5)
        filter_outline_box = wx.BoxSizer(wx.HORIZONTAL)
        filter_outline_box.Add(wx.StaticText(p, -1, 'Outline Color'), 1)
        self.outline_color = ColorButton(p, -1, colorbox_size)
        filter_outline_box.Add(self.outline_color, 0)
        filter_settings_box.Add(filter_outline_box, 0, wx.EXPAND | wx.ALL,
                                5)
        advance_button = wx.Button(p, -1, 'Advance')
        advance_button.Bind(wx.EVT_BUTTON, self.on_advance_button)
        filter_settings_box.Add(advance_button, 0, wx.EXPAND | wx.ALL, 5)
        postprocessing_button = wx.Button(p, -1, 'Postprocessing')
        postprocessing_button.Bind(wx.EVT_BUTTON, self.on_postprocessing_button)
        filter_settings_box.Add(postprocessing_button, 0, wx.EXPAND | wx.ALL, 5)
        pixel_color_box = wx.StaticBoxSizer(wx.HORIZONTAL, p, 'Pixel Color')
        self.pixel_color_text = wx.StaticText(p, -1, 'R255 G255 B255')
        pixel_color_box.Add(self.pixel_color_text, 1,
                            wx.EXPAND | wx.ALL, 5)
        self.pixel_color_button = ColorButton(p, -1, colorbox_size, True)
        pixel_color_box.Add(self.pixel_color_button, 0,
                            wx.EXPAND | wx.ALL, 5)
        box_left.Add(self.zoom_box, 0, wx.EXPAND)
        box_left.Add(region_box, 0, wx.EXPAND)
        box_left.Add(filter_settings_box, 0, wx.EXPAND)
        box_left.Add(pixel_color_box, 0, wx.EXPAND)
        start_button = wx.Button(p, -1, 'Start')
        start_button.Bind(wx.EVT_BUTTON, self.on_start_button)
        box_left.Add(start_button, 0, wx.EXPAND | wx.ALL, 5)
        box_right = wx.StaticBoxSizer(wx.VERTICAL, p, 'Preview')
        _, self.img = video.read()
        self.current_frame = 0
        cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB, self.img)
        wximg = wx.ImageFromBuffer(self.img.shape[1], self.img.shape[0],
                                   self.img)
        self.preview_bitmap = wx.StaticBitmap(p, -1,
                                              bitmap=wximg.ConvertToBitmap())
        self.preview_bitmap.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.preview_bitmap.Bind(wx.EVT_RIGHT_DOWN, self.on_right_down)
        self.preview_bitmap.Bind(wx.EVT_MOTION, self.on_motion)
        preview_box = wx.BoxSizer()
        preview_box.Add(self.preview_bitmap, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        box_right.Add(preview_box, 1, wx.ALIGN_CENTER)
        if frame_count > 1:
            self.preview_slider = wx.Slider(p, -1, maxValue=frame_count - 1)
            self.preview_slider.Bind(wx.EVT_SLIDER, self.on_slider)
            box_right.Add(self.preview_slider, 0,
                          wx.EXPAND)
        box = wx.BoxSizer(wx.HORIZONTAL)
        box.Add(box_left, 0, wx.ALL, 5)
        box.Add(box_right, 1, wx.EXPAND | wx.TOP | wx.RIGHT | wx.BOTTOM, 5)
        p.SetSizer(box)
        box.Fit(p)
        self.Fit()
        self.status = self.STATUS_NONE
        self.update_filter()

    def on_left_down(self, event):
        x, y = self.preview_bitmap.ScreenToClient(wx.GetMousePosition())
        r = 100 / self.zoom_spin.GetValue()
        x, y = x * r, y * r
        if x < self.right_spin.GetValue():
            self.left_spin.SetValue(min(x, self.left_spin.GetMax()))
        else:
            self.right_spin.SetValue(x)
        if y < self.bottom_spin.GetValue():
            self.top_spin.SetValue(min(y, self.top_spin.GetMax()))
        else:
            self.bottom_spin.SetValue(y)
        self.status = self.STATUS_REGION
        self.update_preview()

    def on_right_down(self, event):
        x, y = self.preview_bitmap.ScreenToClient(wx.GetMousePosition())
        r = 100 / self.zoom_spin.GetValue()
        x, y = x * r, y * r
        if x > self.left_spin.GetValue():
            self.right_spin.SetValue(max(x, self.right_spin.GetMin()))
        else:
            self.left_spin.SetValue(x)
        if y > self.top_spin.GetValue():
            self.bottom_spin.SetValue(max(y, self.bottom_spin.GetMin()))
        else:
            self.top_spin.SetValue(y)
        self.status = self.STATUS_REGION
        self.update_preview()

    def on_motion(self, event):
        x, y = self.preview_bitmap.ScreenToClient(wx.GetMousePosition())
        r = self.zoom_spin.GetValue() / 100
        if r != 1:
            img = cv2.resize(self.img, (0, 0), self.img, r, r,
                             cv2.INTER_LANCZOS4 if r < 1 else cv2.INTER_NEAREST)
        else:
            img = self.img
        color = img[y, x]
        self.pixel_color_text.SetLabelText(
            f'R{color[0]} G{color[1]} B{color[2]}')
        self.pixel_color_button.SetBackgroundColour(color)

    def on_spinctrl(self, event):
        self.update_preview()
        self.p.Layout()

    def on_region_spinctrl(self, event):
        if self.status != self.STATUS_NONE:
            self.update_preview()

    def on_slider(self, event):
        frame = self.preview_slider.GetValue()
        if frame == self.current_frame:
            return
        elif frame != self.current_frame + 1:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame)

        _, self.img = self.video.read()
        cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB, self.img)
        self.current_frame = frame
        self.update_preview()

    def on_filter_checkbox(self, event):
        if self.filter_checkbox.GetValue():
            self.status = self.STATUS_FILTER
        else:
            self.status = self.STATUS_NONE
        self.update_preview()

    def on_filter_detail(self, event):
        if self.filter_checkbox.GetValue():
            self.status = self.STATUS_FILTER
        self.update_filter()

    def on_advance_button(self, event):
        self.advance.Show()

    def on_postprocessing_button(self, event):
        self.postprocessing.Show()

    def update_filter(self):
        subtitle_color = self.subtitle_color.GetBackgroundColour()[:3]
        if self.style_choice.GetSelection():
            outline = self.advance.outline.get_filter()
            outline.ref = self.outline_color.GetBackgroundColour()[:3]
            pass1 = self.advance.pass1.get_filter()
            pass1.ref = subtitle_color
        else:
            outline = None
            pass1 = None
        final = self.advance.final.get_filter()
        final.ref = subtitle_color
        self.filter = Filter(self.acolor_checkbox.GetValue(),
                             outline, pass1, final)
        if self.status == self.STATUS_FILTER:
            self.update_preview()

    def update_preview(self):
        t, b = self.top_spin.GetValue(), self.bottom_spin.GetValue()
        l, r = self.left_spin.GetValue(), self.right_spin.GetValue()
        self.top_spin.SetMax(b - 10)
        self.bottom_spin.SetMin(t + 10)
        self.left_spin.SetMax(r - 10)
        self.right_spin.SetMin(l + 10)
        if self.status == self.STATUS_REGION:
            img = self.img.copy()
            img[t:b + 1, l:r + 1, :] = ((img[t:b + 1, l:r + 1, :]
                                         + (215, 227, 241)) / 2)
        elif self.status == self.STATUS_FILTER:
            img = self.img.copy()
            img[t:b + 1, l:r + 1, :] = self.filter(img[t:b + 1, l:r + 1, :])
        else:
            img = self.img
        r = self.zoom_spin.GetValue() / 100
        if r != 1:
            img = cv2.resize(img, (0, 0), img, r, r,
                             cv2.INTER_LANCZOS4 if r < 1 else cv2.INTER_NEAREST)
        self.preview_bitmap.SetBitmap(
            wx.ImageFromBuffer(img.shape[1], img.shape[0],
                               img).ConvertToBitmap())

    def on_start_button(self):
        t, b = self.top_spin.GetValue(), self.bottom_spin.GetValue()
        l, r = self.left_spin.GetValue(), self.right_spin.GetValue()
        self.Hide()
        self.filter.debug = False
        esr(t, b, l ,r, self.filter)


def esr(t, b, l, r, filter):
    ...


if __name__ == '__main__':
    app = wx.App()
    fd = wx.FileDialog(None)
    if fd.ShowModal() != wx.ID_OK:
        exit(1)
    filename = fd.GetPath()
    print(f'Loading Video "{filename}" ...')
    v = cv2.VideoCapture(filename)
    q = sp.run(
        ['ffmpeg', '-i', filename, '-map', '0:v:0', '-c', 'copy', '-f',
         'null',
         '-'], stdin=sp.DEVNULL, stdout=sp.DEVNULL, stderr=sp.PIPE)
    fc = int(q.stderr[q.stderr.rfind(b'frame=') + 6:].split(None, 1)[0])
    f = FilterFrame(None, -1, v, fc)
    f.Show()
    exit(app.MainLoop())
