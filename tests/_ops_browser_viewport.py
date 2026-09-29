"""Set a measured CSS viewport without changing the production screenshot tool.

Firefox's headless window cannot be narrower than 450 physical pixels.  A 2x
content zoom plus a window twice the requested width gives an exact 320/390 CSS
pixel viewport.  Integer zoom avoids Firefox's fractional zoom quantisation.
The helper restores 1x zoom when returning to a desktop-sized viewport.
"""
import time


def set_css_viewport(browser, width: int, height: int = 844) -> dict:
    """Resize, assert the actual content viewport, and return its measurements.

    ``browser`` is a ``wss_deploy.devshot.Browser`` instance.  This changes only
    browser geometry and content zoom, never page CSS or viewport metadata.
    Returned ``clientWidth`` excludes a classic vertical scrollbar, if present;
    a full-page screenshot may therefore be narrower than ``innerWidth``.
    """
    if width < 225 or height < 300:
        raise ValueError("Unsupported test viewport: minimum 225 by 300 CSS pixels")
    zoom = 2 if width < 450 else 1
    browser.cmd("Marionette:SetContext", {"value": "chrome"})
    try:
        browser.js("window.gBrowser.selectedBrowser.fullZoom=arguments[0]", [zoom])
    finally:
        browser.cmd("Marionette:SetContext", {"value": "content"})

    outer_width, outer_height = width * zoom, height * zoom + 180
    metrics = {}
    for _ in range(6):
        browser.resize(outer_width, outer_height)
        time.sleep(0.15)
        metrics = browser.js("return {innerWidth, innerHeight, "
                             "clientWidth:document.documentElement.clientWidth, "
                             "scrollWidth:document.documentElement.scrollWidth, "
                             "devicePixelRatio}")
        if metrics["innerWidth"] == width and metrics["innerHeight"] == height:
            metrics.update(requested_width=width, requested_height=height, zoom=zoom)
            return metrics
        # Browser chrome height can change when a toolbar wraps at a narrow
        # width.  Correct from measured content dimensions, not assumed chrome.
        outer_width += (width - metrics["innerWidth"]) * zoom
        outer_height += (height - metrics["innerHeight"]) * zoom
    raise AssertionError(f"Requested {width}x{height} CSS viewport; measured {metrics}")
