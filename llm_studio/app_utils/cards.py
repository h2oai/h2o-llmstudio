from typing import List, Optional

from h2o_wave import ui


def header_zone() -> ui.Zone:
    """Returns the header zone"""

    zone = ui.zone(
        "header",
        size="80px",
    )

    return zone


def navigation_zone() -> ui.Zone:
    """Returns the navigation zone"""

    zone = ui.zone(
        "navigation",
        size="max(13%, 180px)",
        zones=[
            ui.zone(name="nav", size="100%"),
        ],
    )

    return zone


def card_zones(mode: Optional[str] = "full") -> List[ui.Zone]:
    """Specifies for certain modes the layout zones

    Args:
        mode: mode for layout zones

    Returns:
        List of zones

    """

    if mode in ["full", "experiment_start"]:
        zones = [
            header_zone(),
            ui.zone(
                "body",
                size="1",
                direction=ui.ZoneDirection.ROW,
                zones=[
                    navigation_zone(),
                    ui.zone(
                        "content_all",
                        direction=ui.ZoneDirection.COLUMN,
                        size="min(calc(100% - 180px), 87%)",
                        zones=[
                            ui.zone(
                                "content",
                                size="calc(100vh - 160px)",
                                direction=ui.ZoneDirection.ROW,
                            ),
                            ui.zone("footer", size="80px"),
                        ],
                    ),
                ],
            ),
        ]
    elif mode == "error":
        zones = [
            header_zone(),
            ui.zone(
                "body",
                size="1",
                direction=ui.ZoneDirection.ROW,
                zones=[
                    navigation_zone(),
                    ui.zone(
                        "content_all",
                        direction=ui.ZoneDirection.COLUMN,
                        size="min(calc(100% - 180px), 87%)",
                        zones=[
                            ui.zone(
                                "content",
                                size="calc(100vh - 80px)",
                                direction=ui.ZoneDirection.ROW,
                            ),
                        ],
                    ),
                ],
            ),
        ]

    elif mode == "home":
        zones = [
            header_zone(),
            ui.zone(
                "body",
                size="calc(100vh - 80px)",
                direction=ui.ZoneDirection.ROW,
                zones=[
                    navigation_zone(),
                    ui.zone(
                        "content_all",
                        direction=ui.ZoneDirection.COLUMN,
                        size="min(calc(100% - 180px), 87%)",
                        zones=[
                            ui.zone(
                                "content",
                                size="370px",
                                direction=ui.ZoneDirection.ROW,
                            ),
                            ui.zone(
                                "expander",
                                size="0",
                                direction=ui.ZoneDirection.ROW,
                            ),
                            ui.zone(
                                "tables",
                                size="0",
                                direction=ui.ZoneDirection.ROW,
                                zones=[
                                    ui.zone(name="datasets", size="40%"),
                                    ui.zone(name="experiments", size="60%"),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ]

    elif mode in [
        "experiment/display/charts",
        "experiment/compare/charts",
    ]:
        zones = [
            header_zone(),
            ui.zone(
                "body",
                size="1",
                direction=ui.ZoneDirection.ROW,
                zones=[
                    navigation_zone(),
                    ui.zone(
                        "content_all",
                        direction=ui.ZoneDirection.COLUMN,
                        size="min(calc(100% - 180px), 87%)",
                        zones=[
                            ui.zone("nav2", size="62px"),
                            ui.zone(
                                "first_row",
                                size="max(calc((100vh - 222px)*0.5), 300px)",
                                direction=ui.ZoneDirection.ROW,
                                zones=[
                                    ui.zone("top_left", size="50%"),
                                    ui.zone("top_right", size="50%"),
                                ],
                            ),
                            ui.zone(
                                "second_row",
                                size="max(calc((100vh - 222px)*0.5), 300px)",
                                direction=ui.ZoneDirection.ROW,
                                zones=[
                                    ui.zone("bottom_left", size="50%"),
                                    ui.zone("bottom_right", size="50%"),
                                ],
                            ),
                            ui.zone("footer", size="80px"),
                        ],
                    ),
                ],
            ),
        ]

    elif mode in [
        "experiment/display/chat",
    ]:
        zones = [
            header_zone(),
            ui.zone(
                "body",
                size="1",
                direction=ui.ZoneDirection.ROW,
                zones=[
                    navigation_zone(),
                    ui.zone(
                        "content_all",
                        direction=ui.ZoneDirection.COLUMN,
                        size="min(calc(100% - 180px), 87%)",
                        zones=[
                            ui.zone("nav2", size="62px"),
                            ui.zone(
                                "first",
                                size="calc((100vh - 222px)*0.65)",
                                direction=ui.ZoneDirection.ROW,
                            ),
                            ui.zone(
                                "second",
                                size="calc((100vh - 222px)*0.35)",
                                direction=ui.ZoneDirection.ROW,
                            ),
                            ui.zone("footer", size="80px"),
                        ],
                    ),
                ],
            ),
        ]
    elif mode in ["experiment/display/summary"]:
        zones = [
            header_zone(),
            ui.zone(
                "body",
                size="1",
                direction=ui.ZoneDirection.ROW,
                zones=[
                    navigation_zone(),
                    ui.zone(
                        "content_all",
                        direction=ui.ZoneDirection.COLUMN,
                        size="min(calc(100% - 180px), 87%)",
                        zones=[
                            ui.zone("nav2", size="62px"),
                            ui.zone(
                                "first",
                                size="235px",
                                direction=ui.ZoneDirection.ROW,
                            ),
                            ui.zone(
                                "second",
                                size="235px",
                                direction=ui.ZoneDirection.ROW,
                            ),
                            ui.zone(
                                "third",
                                size="max(calc(100vh - 692px), 400px)",
                                direction=ui.ZoneDirection.ROW,
                            ),
                            ui.zone("footer", size="80px"),
                        ],
                    ),
                ],
            ),
        ]
    elif mode in ["dataset/display/statistics"]:
        zones = [
            header_zone(),
            ui.zone(
                "body",
                size="1",
                direction=ui.ZoneDirection.ROW,
                zones=[
                    navigation_zone(),
                    ui.zone(
                        "content_all",
                        direction=ui.ZoneDirection.COLUMN,
                        size="min(calc(100% - 180px), 87%)",
                        zones=[
                            ui.zone("nav2", size="62px"),
                            ui.zone(
                                "first",
                                size="max(calc(0.33*(100vh - 222px)), 400px)",
                                direction=ui.ZoneDirection.ROW,
                            ),
                            ui.zone(
                                "second",
                                size="max(calc(0.33*(100vh - 222px)), 400px)",
                                direction=ui.ZoneDirection.ROW,
                            ),
                            ui.zone(
                                "third",
                                size="max(calc(0.34*(100vh - 222px)), 200px)",
                                direction=ui.ZoneDirection.ROW,
                            ),
                            ui.zone("footer", size="80px"),
                        ],
                    ),
                ],
            ),
        ]
    elif mode in [
        "experiment/compare/config",
        "experiment/display/train_data_insights",
        "experiment/display/validation_prediction_insights",
        "experiment/display/config",
        "experiment/display/logs",
        "dataset/display/data",
        "dataset/display/visualization",
        "dataset/display/summary",
    ]:
        zones = [
            header_zone(),
            ui.zone(
                "body",
                size="1",
                direction=ui.ZoneDirection.ROW,
                zones=[
                    navigation_zone(),
                    ui.zone(
                        "content_all",
                        direction=ui.ZoneDirection.COLUMN,
                        size="min(calc(100% - 180px), 87%)",
                        zones=[
                            ui.zone("nav2", size="62px"),
                            ui.zone(
                                "first",
                                size="calc(100vh - 222px)",
                                direction=ui.ZoneDirection.ROW,
                            ),
                            ui.zone("footer", size="80px"),
                        ],
                    ),
                ],
            ),
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return zones


def card_wait(msg: str, box: str) -> ui.FormCard:
    """Return a form card for displaying waiting status

    Args:
        msg: message to display
        box: box for card

    Returns:
        Form card

    """

    card = ui.form_card(box=box, items=[ui.progress(label=msg)])

    return card
