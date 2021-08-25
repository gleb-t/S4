import os
from numbers import Real
from typing import *
from typing import List

import numpy as np
import cairo


class FontExtents:

    def __init__(self, ascent, descent, height, max_x_advance, max_y_advance):
        self.ascent = ascent
        self.descent = descent
        self.height = height
        self.max_x_advance = max_x_advance
        self.max_y_advance = max_y_advance


class CairoFileSurface:
    """
    Wraps a PyCairo surface objects, creating and appropriate surface for a given file type, writes it afterwards.
    Implements the enter/exit interface for the "with" statement.
    """

    def __init__(self, filepath: str, figureSize: Tuple[float, float]):
        self.filepath = filepath
        outputFileExtension = os.path.splitext(filepath)[1]
        if outputFileExtension == '.pdf':
            self.surface = cairo.PDFSurface(filepath, *figureSize)
            if hasattr(self.surface, 'set_metadata'):  # The method went missing, no time to figure out why.
                self.surface.set_metadata(cairo.PDFMetadata.CREATE_DATE, '2000-01-01T00:00:00')  # Deterministic output.
        elif outputFileExtension == '.png':
            self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(figureSize[0]), int(figureSize[1]))
        else:
            raise RuntimeError("Unsupported output extension: '{}'".format(outputFileExtension))

    def get_context(self) -> cairo.Context:
        return cairo.Context(self.surface)

    def __enter__(self):
        self.surface.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # PDF surfaces are written automatically, PNGs need to be saved explicitly,
        if isinstance(self.surface, cairo.ImageSurface):
            self.surface.write_to_png(self.filepath)
            self.surface.__exit__(exc_type, exc_val, exc_tb)


class CairoNumpySurface:

    def __init__(self, figureSize: Tuple[float, float]):

        self.figureSize = figureSize
        self.surfaceFormat = cairo.FORMAT_ARGB32  # For some reason, couldn't get data from 'A8' format correctly.
        self.surface = cairo.ImageSurface(self.surfaceFormat, int(figureSize[0]), int(figureSize[1]))

    def __enter__(self):
        self.surface.__enter__()

        return self

    def get_context(self) -> cairo.Context:
        return cairo.Context(self.surface)

    def get_numpy(self) -> np.ndarray:
        buffer = self.surface.get_data()
        channelNumber = 4 if self.surfaceFormat == cairo.FORMAT_ARGB32 else 1
        data = np.ndarray(shape=(self.figureSize[1], self.figureSize[0], channelNumber),
                          dtype=np.uint8,
                          buffer=buffer)

        return data

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.surface.__exit__(exc_type, exc_val, exc_tb)


def cairo_show_text_centered(cr: cairo.Context, text: str, x: Real, y: Real):
    xBearing, yBearing, width, height, xAdvance, yAdvance = cr.text_extents(text)
    x -= width / 2 + xBearing
    y -= height / 2 + yBearing

    cr.move_to(x, y)
    cr.show_text(text)


def cairo_show_text_centered_with_superscript(cr: cairo.Context, 
                                              textNorm: str, textSuper: str,
                                              fontSizeNormal: Real, fontSizeSuper: Real,
                                              x: Real, y: Real):
    cr.save()
    cr.set_font_size(fontSizeNormal)
    teNorm = cr.text_extents(textNorm)
    feNorm = FontExtents(*cr.font_extents())
    cr.set_font_size(fontSizeSuper)
    teSuper = cr.text_extents(textSuper)
    feSuper = FontExtents(*cr.font_extents())
    padSuper = fontSizeSuper / 4  # A hack to add more space.

    shiftX = -teNorm.width / 2 - teNorm.x_bearing - teSuper.width / 2
    shiftY = teNorm.height / 2 - feNorm.descent

    # cr.rectangle(x, y, 0.5, 0.5)
    # cr.stroke()

    cr.set_font_size(fontSizeNormal)
    cr.move_to(x + shiftX, y + shiftY)
    cr.show_text(textNorm)

    cr.set_font_size(fontSizeSuper)
    cr.move_to(x + teNorm.width / 2 - teSuper.x_bearing,
               y - teNorm.height / 2 + feNorm.descent - teSuper.height / 2 + feSuper.descent)
    cr.show_text(textSuper)

    cr.restore()


def render_image_into_cairo(cr: cairo.Context, image: np.ndarray,
                            x: float, y: float, width: float, height: float):
    """

    :param cr:
    :param image: A numpy array of shape (height, width, 4), with uint8 RGBA channels.
    :param x:
    :param y:
    :param width:
    :param height:
    :return:
    """

    with image_rgba_uint8_to_cairo(image) as imageSurface:
        cr.save()

        origWidth, origHeight = imageSurface.get_width(), imageSurface.get_height()
        scale = min(width / origWidth, height / origHeight)
        scaledWidth, scaledHeight = origWidth * scale, origHeight * scale
        offsetX, offsetY = (width - scaledWidth) / 2, (height - scaledHeight) / 2

        pattern = cairo.SurfacePattern(imageSurface)
        # We need to align the cairo source pattern with the rectangle that we're going to draw.
        matrix = cairo.Matrix()
        matrix.scale(1 / scale, 1 / scale)
        matrix.translate(-(x + offsetX), -(y + offsetY))
        pattern.set_matrix(matrix)
        cr.set_source(pattern)

        cr.rectangle(x + offsetX, y + offsetY, scaledWidth, scaledHeight)
        cr.fill()

        cr.restore()

        return x + offsetX, y + offsetY, x + offsetX + scaledWidth, y + offsetY + scaledHeight


def render_colorbar(cr,
                    colormap: Callable,
                    labels: Tuple[str, str],
                    x: float, y: float,
                    width: float, height: float, fontSize: float,
                    isVertical: bool = True,
                    margins: Optional[Tuple[float, float]] = None):

    if not margins and isVertical:
        margins = (fontSize / 2, fontSize * 1.5)

    if not isVertical:
        # In the vertical case, extend the margins with the text size.
        marginsDef = margins or (fontSize / 2, fontSize / 2)

        cr.set_font_size(fontSize)
        maxTextWidth = max(cr.text_extents(labels[0]).width, cr.text_extents(labels[1]).width)
        margins = (marginsDef[0] + maxTextWidth, marginsDef[1])

    marginX, marginY = margins
    rectWidth = width - 2 * marginX
    rectHeight = height - 2 * marginY

    cr.save()
    cr.translate(x, y)

    if isVertical:
        gradient = cairo.LinearGradient(marginX, marginY, marginX, rectHeight + marginY)
    else:
        gradient = cairo.LinearGradient(marginX, marginY, rectWidth + marginX, marginY)
    gradientResolution = 100
    for i in range(gradientResolution):
        t = i / (gradientResolution - 1)
        color = colormap(t)
        if isVertical:
            t = 1 - t  # Invert the Y axis.
        gradient.add_color_stop_rgba(t, *color)
    cr.set_source(gradient)
    cr.rectangle(marginX, marginY, rectWidth, rectHeight)
    cr.fill()

    cr.set_source_rgba(0.0, 0.0, 0.0, 1.0)
    cr.set_font_size(fontSize)

    if isVertical:
        cairo_show_text_centered(cr, labels[0], width / 2, height - marginY / 2)
        cairo_show_text_centered(cr, labels[1], width / 2, marginY / 2)
    else:
        cairo_show_text_centered(cr, labels[0], marginX / 2, height / 2)
        cairo_show_text_centered(cr, labels[1], width - marginX / 2, height / 2)

    cr.restore()


def image_rgba_uint8_to_cairo(imageData: np.ndarray):

    assert imageData.dtype == np.uint8
    assert len(imageData.shape) == 3  # height x width x channels
    assert imageData.shape[2] == 4    # RGBA

    # Flip the Y axis (first axis in C-order) to point bottom-up.
    imageData = np.flip(imageData, axis=0)

    # Convert RGBA to ARGB and reinterpret as a uint32.
    # (uint32 is little-endian, so we actually need BGRA)
    imageBGR = np.flip(imageData[:, :, :3], axis=2)
    imageA = imageData[:, :, 3][..., np.newaxis]
    # Copy to guarantee memory-continuity.
    imageDataConverted = np.concatenate((imageBGR, imageA), axis=2).copy().view('<u4')
    return cairo.ImageSurface.create_for_data(imageDataConverted, cairo.FORMAT_ARGB32,
                                              imageData.shape[1], imageData.shape[0])


def text_path_multiline(cr: cairo.Context,
                        text: str,
                        x: float,
                        y: float,
                        lineWidth: float,
                        lineSpacing: float,
                        maxLines: int = 0,
                        ellipsis: str = ' ..'):

    def _get_words_fit(words: List[str], lineWidth: float):
        wordsThatFit = 0
        while True:
            textLine = ' '.join(words[0:wordsThatFit + 1])
            if wordsThatFit < len(words):
                textLine += ellipsis

            _, _, width, _, _, _ = cr.text_extents(textLine)
            if width > lineWidth or wordsThatFit >= len(words):
                break
            else:
                wordsThatFit += 1

        return wordsThatFit

    words = text.split(' ')
    iLine = 0
    wordsDrawn, wordsTotal = 0, len(words)
    while len(words) > 0 and iLine < maxLines:
        wordsToDraw = _get_words_fit(words, lineWidth)
        textLine = ' '.join(words[0:wordsToDraw])
        if iLine == maxLines - 1 and wordsDrawn + wordsToDraw < wordsTotal:
            textLine += ellipsis

        _, _, width, height, _, _ = cr.text_extents(textLine)
        assert width <= lineWidth

        cr.move_to(x, y + iLine * lineSpacing + height)
        cr.text_path(textLine)
        words = words[wordsToDraw:]
        wordsDrawn += wordsToDraw
        iLine += 1
