import functools
import json
import traceback

import requests
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def sc_send(title, desp="", options=None):
    if options is None:
        options = {
            "channel": 9,
        }

    # Prepare the payload to send to the SCT API
    payload = {
        "title": title,
        "desp": desp,
        "short": options.get("short", ""),
        "noip": options.get("noip", ""),
        "channel": options["channel"],
        "openid": options.get("openid", ""),
    }

    # The Server URL
    sendkey = config["push_key"]
    url = f"https://sctapi.ftqq.com/{sendkey}.send"

    # Set headers for content type as JSON
    headers = {"Content-Type": "application/json;charset=utf-8"}

    # Send the POST request to API
    response = requests.post(url, data=json.dumps(payload), headers=headers)

    # Check for successful response
    if response.status_code != 200:
        raise Exception(f"Failed to send notification: {response.text}")

    # Return the result as a JSON object
    result = response.json()
    return result


def notify_on_exception(title_prefix="Exception occurred"):
    """
    A decorator to wrap any function. If an exception occurs, it sends a notification
    using the `sc_send` function.

    Args:
        sendkey (str): The sendkey used for sending notifications.
        title_prefix (str): Prefix for the notification title.

    Returns:
        function: The wrapped function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Capture exception details
                exc_info = traceback.format_exc()

                # Notification title and description
                title = f"{title_prefix}: {func.__name__}"
                description = f"Exception: {str(e)}\n\nTraceback:\n{exc_info}"

                # Send the notification
                try:
                    sc_send(config["push_key"], title, description)

                except Exception as notify_error:
                    print(f"Failed to send notification: {notify_error}")

                # Re-raise the original exception after notifying
                raise

        return wrapper

    return decorator


@notify_on_exception()
def run_function(x, y):
    return x / y


if __name__ == "__main__":
    # run_function(10, 0)
    sc_send(
        "Test Notification",
        "This is a test message to verify the notification system.",
    )


