from napari.utils.notifications import (
    Notification,
    NotificationSeverity,
    notification_manager,
)


def napari_notification(msg: str, severity: NotificationSeverity = NotificationSeverity.INFO) -> None:
    """Display a napari notification within the napari viewer with a given severity."""
    notification_ = Notification(msg, severity=severity)
    notification_manager.dispatch(notification_)
