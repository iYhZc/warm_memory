"""Root-group fallbacks without registering a second command of the same name."""
from __future__ import annotations

import re


def _tail(text, names):
    """Match the SDK's whole-word names, preserving the argument's original text."""
    for name in sorted(names, key=len, reverse=True):
        pattern = r"\s+".join(re.escape(word) for word in name.split())
        if not pattern:
            continue
        match = re.match(r"^\s*" + pattern + r"(?=\s|$)", text)
        if match:
            return text[match.end():].strip()
    return None


class BareGroupGuard:
    """Let the fallback handle a bare group instead of the SDK's missing-args error."""

    def __init__(self, group):
        self.group = group

    def filter(self, event, config):
        return not self.group.equals(event.get_message_str().strip())


class GroupFallbackFilter:
    def __init__(self, group):
        self.group = group

    def match(self, event, config, handlers):
        if not getattr(event, "is_at_or_wake_command", False):
            return False
        text = event.get_message_str()
        rest = _tail(text, self.group.get_complete_command_names())
        if rest is None:
            return False
        # A registered child owns all its arguments, including invalid ones.
        # Do not invoke CommandFilter.filter: it parses and mutates the event.
        for child in self.group.sub_command_filters:
            name = getattr(child, "command_name", getattr(child, "group_name", ""))
            if _tail(rest, [name, *getattr(child, "alias", ())]) is not None:
                return False
        owners = [handler for handler in handlers
                  if self.group in handler.event_filters]
        if len(owners) != 1 or not owners[0].enabled:
            event.stop_event()
            return False
        owner = owners[0]
        # Read the current group controls on every event (WebUI can change them).
        # Fail closed on missing/incompatible controls; never bypass a restriction.
        try:
            for control in owner.event_filters:
                if control is not self.group and not control.filter(event, config):
                    event.stop_event()
                    return False
            for control in self.group.custom_filter_list:
                if not isinstance(control, BareGroupGuard) and not control.filter(event, config):
                    event.stop_event()
                    return False
        except Exception:
            event.stop_event()
            return False
        event.set_extra("parsed_params", {"rest": rest})
        return True


def command_group(name):
    """Keep the native group identity, child registration and WebUI controls."""
    from astrbot.api.event import filter

    def decorate(handler):
        registered = filter.command_group(name)(handler)
        registered.parent_group.add_custom_filter(BareGroupGuard(registered.parent_group))
        return registered

    return decorate


def fallback_for(registered):
    """Register only a custom listener filter, invisible to command conflict scans."""
    from astrbot.api.event import filter

    class Fallback:
        def __init__(self, raise_error=True):
            self.route = GroupFallbackFilter(registered.parent_group)

        def filter(self, event, config):
            from astrbot.core.star.star_handler import star_handlers_registry

            return self.route.match(event, config, star_handlers_registry)

    return filter.custom_filter(Fallback)
