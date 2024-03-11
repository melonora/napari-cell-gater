from napari.utils.notifications import (
    Notification,
    NotificationSeverity,
    notification_manager,
)


def napari_notification(msg, severity=NotificationSeverity.INFO):
    notification_ = Notification(msg, severity=severity)
    notification_manager.dispatch(notification_)
