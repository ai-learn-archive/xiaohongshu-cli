"""Reading commands: search, read, comments, sub-comments, user, user-posts, feed, hot, topics, search-user."""

import re

import click

from ..client_mixins import build_search_filters, build_search_geo
from ..command_normalizers import normalize_paged_notes
from ..cookies import cache_note_context
from ..formatter import (
    maybe_print_structured,
    print_info,
    render_comments,
    render_feed,
    render_note,
    render_search_results,
    render_topics,
    render_user_info,
    render_user_posts,
    render_users,
)
from ..note_refs import resolve_note_reference, save_index_from_items, save_index_from_notes
from ._common import exit_for_error, handle_command, run_client_action, structured_output_options

# ─── Token propagation ─────────────────────────────────────────────────────

def _cache_tokens_from_items(data: dict, *, xsec_source: str) -> None:
    """Auto-cache xsec_token from search/feed API results.

    Each note item may carry its own xsec_token bound to the source
    (search, feed, explore).  Caching them lets a subsequent
    `xhs read <note_id>` use the correct token automatically.
    """
    for item in data.get("items", []):
        note_card = item.get("note_card", {})
        note_id = item.get("id", note_card.get("note_id", ""))
        token = item.get("xsec_token", note_card.get("xsec_token", ""))
        if note_id and token:
            cache_note_context(note_id, token, xsec_source)

# ─── Sort mapping ────────────────────────────────────────────────────────────

SORT_MAP = {
    "general": "general",
    "popular": "popularity_descending",
    "latest": "time_descending",
}

SORT_TAG_MAP = {
    "general": "综合",
    "popular": "最多点赞",
    "latest": "最新",
}

TYPE_MAP = {
    "all": 0,
    "不限": 0,
    "video": 1,
    "视频": 1,
    "image": 2,
    "图文": 2,
}

TYPE_TAG_MAP = {
    "all": "不限",
    "不限": "不限",
    "video": "视频",
    "视频": "视频",
    "image": "图文",
    "图文": "图文",
}

TIME_TAG_MAP = {
    "all": "不限",
    "不限": "不限",
    "day": "一天内",
    "一天内": "一天内",
    "week": "一周内",
    "一周内": "一周内",
    "half-year": "半年内",
    "半年内": "半年内",
}

RANGE_TAG_MAP = {
    "all": "不限",
    "不限": "不限",
    "seen": "已看过",
    "已看过": "已看过",
    "unseen": "未看过",
    "未看过": "未看过",
    "followed": "已关注",
    "已关注": "已关注",
}

DISTANCE_TAG_MAP = {
    "all": "不限",
    "不限": "不限",
    "city": "同城",
    "同城": "同城",
    "nearby": "附近",
    "附近": "附近",
}

SEARCH_DEFAULT_OPTIONS = {
    "sort": "general",
    "note_type": "all",
    "time_filter": "all",
    "search_range": "all",
    "distance_filter": "all",
}

SEARCH_TOKEN_ALIASES = {
    "general": ("sort", "general"),
    "综合": ("sort", "general"),
    "popular": ("sort", "popular"),
    "最多点赞": ("sort", "popular"),
    "latest": ("sort", "latest"),
    "最新": ("sort", "latest"),
    "video": ("note_type", "video"),
    "视频": ("note_type", "视频"),
    "image": ("note_type", "image"),
    "图文": ("note_type", "图文"),
    "day": ("time_filter", "day"),
    "一天内": ("time_filter", "一天内"),
    "week": ("time_filter", "week"),
    "一周内": ("time_filter", "一周内"),
    "half-year": ("time_filter", "half-year"),
    "半年内": ("time_filter", "半年内"),
    "seen": ("search_range", "seen"),
    "已看过": ("search_range", "已看过"),
    "unseen": ("search_range", "unseen"),
    "未看过": ("search_range", "未看过"),
    "followed": ("search_range", "followed"),
    "已关注": ("search_range", "已关注"),
    "city": ("distance_filter", "city"),
    "同城": ("distance_filter", "同城"),
    "nearby": ("distance_filter", "nearby"),
    "附近": ("distance_filter", "附近"),
}

SEARCH_OPTION_LABELS = {
    "sort": "排序",
    "note_type": "笔记类型",
    "time_filter": "发布时间",
    "search_range": "搜索范围",
    "distance_filter": "位置距离",
}

DISTANCE_LOCATION_SUFFIXES = ("同城", "附近")


def _tokenize_search_sorts(value: str) -> list[str]:
    return [token for token in re.split(r"[\s、,，]+", value.strip()) if token]


def _split_distance_location_token(token: str) -> tuple[str, str] | None:
    for suffix in DISTANCE_LOCATION_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix):
            location = token[:-len(suffix)].strip()
            if location:
                return location, suffix
    return None


def _parse_search_sorts(value: str) -> tuple[dict[str, str], str | None]:
    resolved: dict[str, str] = {}
    distance_location: str | None = None
    for raw_token in _tokenize_search_sorts(value):
        location_distance = _split_distance_location_token(raw_token)
        if location_distance is not None:
            location, distance_token = location_distance
            option_name, option_value = "distance_filter", distance_token
            if distance_location and distance_location != location:
                raise click.BadParameter(
                    f"位置距离包含多个地点: {distance_location} 和 {location}",
                    param_hint="--sorts",
                )
            distance_location = location
        else:
            token = raw_token.lower()
            mapping = SEARCH_TOKEN_ALIASES.get(token)
            if mapping is None:
                raise click.BadParameter(f"无法识别筛选词: {raw_token}", param_hint="--sorts")
            option_name, option_value = mapping
        previous = resolved.get(option_name)
        if previous and previous != option_value:
            raise click.BadParameter(
                f"{SEARCH_OPTION_LABELS[option_name]}包含多个值: {previous} 和 {raw_token}",
                param_hint="--sorts",
            )
        resolved[option_name] = option_value
    return resolved, distance_location


def _resolve_search_options(
    *,
    sort: str,
    note_type: str,
    sorts: str,
) -> tuple[dict[str, str], str | None, str | None]:
    resolved = {
        "sort": sort,
        "note_type": note_type,
        "time_filter": "all",
        "search_range": "all",
        "distance_filter": "all",
    }
    sorts_values, distance_location = _parse_search_sorts(sorts) if sorts else ({}, None)
    sort_tag: str | None = None

    for option_name, option_value in sorts_values.items():
        existing_value = resolved[option_name]
        default_value = SEARCH_DEFAULT_OPTIONS[option_name]
        if existing_value != default_value and existing_value != option_value:
            raise click.BadParameter(
                f"--sorts 中的 {SEARCH_OPTION_LABELS[option_name]}={option_value} 与显式参数 {existing_value} 冲突",
                param_hint="--sorts",
            )
        resolved[option_name] = option_value
        if option_name == "sort":
            sort_tag = SORT_TAG_MAP[option_value]

    return resolved, sort_tag, distance_location


def _resolve_search_geo(distance_location: str | None) -> str:
    if not distance_location:
        return ""
    try:
        return build_search_geo(distance_location)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--sorts") from None


@click.command()
@click.argument("keyword")
@click.option("--sort", type=click.Choice(["general", "popular", "latest"]), default="general", help="Sort order")
@click.option(
    "--type",
    "note_type",
    type=click.Choice(list(TYPE_MAP.keys()), case_sensitive=False),
    default="all",
    help="Note type / 笔记类型",
)
@click.option(
    "--sorts",
    default="",
    help='Combined Chinese filters, split by space or "、"',
)
@click.option("--page", default=1, help="Page number")
@structured_output_options
@click.pass_context
def search(
    ctx,
    keyword: str,
    sort: str,
    note_type: str,
    sorts: str,
    page: int,
    as_json: bool,
    as_yaml: bool,
):
    """Search notes by keyword."""
    resolved_options, sort_tag, distance_location = _resolve_search_options(
        sort=sort,
        note_type=note_type,
        sorts=sorts,
    )
    geo = _resolve_search_geo(distance_location)

    def _search_action(client):
        result = client.search_notes(
            keyword=keyword,
            page=page,
            sort=SORT_MAP[resolved_options["sort"]],
            note_type=TYPE_MAP[resolved_options["note_type"]],
            geo=geo,
            filters=build_search_filters(
                sort_tag=sort_tag,
                note_type_tag=TYPE_TAG_MAP[resolved_options["note_type"]],
                time_tag=TIME_TAG_MAP[resolved_options["time_filter"]],
                range_tag=RANGE_TAG_MAP[resolved_options["search_range"]],
                distance_tag=DISTANCE_TAG_MAP[resolved_options["distance_filter"]],
            ),
        )
        _cache_tokens_from_items(result, xsec_source="pc_search")
        save_index_from_items(result, xsec_source="pc_search")
        return result

    handle_command(
        ctx,
        action=_search_action,
        render=render_search_results,
        as_json=as_json,
        as_yaml=as_yaml,
    )


@click.command()
@click.argument("id_or_url")
@click.option("--xsec-token", default="", help="Security token (or reuse a cached token for this note)")
@structured_output_options
@click.pass_context
def read(ctx, id_or_url: str, xsec_token: str, as_json: bool, as_yaml: bool):
    """Read a note by ID, URL, or short index."""
    note_id, token, url_source = resolve_note_reference(id_or_url, xsec_token=xsec_token)
    xsec_source = url_source or "pc_feed"
    if token:
        cache_note_context(note_id, token, xsec_source)

    def _read_action(client):
        kwargs = {"xsec_token": token}
        if url_source:
            kwargs["xsec_source"] = url_source
        return client.get_note_detail(note_id, **kwargs)

    handle_command(
        ctx,
        action=_read_action,
        render=render_note,
        as_json=as_json,
        as_yaml=as_yaml,
    )


@click.command()
@click.argument("id_or_url")
@click.option("--cursor", default="", help="Pagination cursor")
@click.option("--xsec-token", default="", help="Security token")
@click.option("--all", "fetch_all", is_flag=True, help="Auto-paginate to fetch ALL comments")
@structured_output_options
@click.pass_context
def comments(ctx, id_or_url: str, cursor: str, xsec_token: str, fetch_all: bool, as_json: bool, as_yaml: bool):
    """View comments on a note by ID, URL, or short index."""
    note_id, token, url_source = resolve_note_reference(id_or_url, xsec_token=xsec_token)
    xsec_source = url_source or "pc_feed"
    if token:
        cache_note_context(note_id, token, xsec_source)

    def _load_comments(client):
        common_kwargs = {"xsec_token": token}
        if url_source:
            common_kwargs["xsec_source"] = url_source
        if fetch_all:
            return client.get_all_comments(note_id, **common_kwargs)
        return client.get_comments(
            note_id,
            cursor=cursor,
            **common_kwargs,
        )

    def _render_comments(data):
        render_comments(data)
        if fetch_all and isinstance(data, dict):
            total = data.get("total_fetched", 0)
            pages = data.get("pages_fetched", 0)
            print_info(f"Fetched {total} comments across {pages} pages")

    try:
        data = run_client_action(ctx, _load_comments)
        if not maybe_print_structured(data, as_json=as_json, as_yaml=as_yaml):
            _render_comments(data)
    except Exception as exc:
        exit_for_error(exc, as_json=as_json, as_yaml=as_yaml)


@click.command()
@click.argument("user_id")
@structured_output_options
@click.pass_context
def user(ctx, user_id: str, as_json: bool, as_yaml: bool):
    """View user profile info."""
    handle_command(
        ctx,
        action=lambda client: client.get_user_info(user_id),
        render=render_user_info,
        as_json=as_json,
        as_yaml=as_yaml,
    )


@click.command("user-posts")
@click.argument("user_id")
@click.option("--cursor", default="", help="Pagination cursor")
@structured_output_options
@click.pass_context
def user_posts(ctx, user_id: str, cursor: str, as_json: bool, as_yaml: bool):
    """List a user's published notes."""
    def _user_posts_action(client):
        data = client.get_user_notes(user_id, cursor=cursor)
        page = normalize_paged_notes(data)
        save_index_from_notes(page["notes"])
        return data

    def _render_user_posts(data):
        page = normalize_paged_notes(data)
        render_user_posts(page["notes"])
        if page["has_more"]:
            print_info(f"More notes available — use --cursor {page['cursor']}")

    handle_command(
        ctx,
        action=_user_posts_action,
        render=_render_user_posts,
        as_json=as_json,
        as_yaml=as_yaml,
    )


@click.command()
@structured_output_options
@click.pass_context
def feed(ctx, as_json: bool, as_yaml: bool):
    """Browse the recommendation feed."""
    def _feed_action(client):
        result = client.get_home_feed()
        _cache_tokens_from_items(result, xsec_source="pc_feed")
        save_index_from_items(result, xsec_source="pc_feed")
        return result

    handle_command(
        ctx,
        action=_feed_action,
        render=render_feed,
        as_json=as_json,
        as_yaml=as_yaml,
    )


@click.command()
@click.argument("keyword")
@structured_output_options
@click.pass_context
def topics(ctx, keyword: str, as_json: bool, as_yaml: bool):
    """Search for topics/hashtags."""
    handle_command(
        ctx,
        action=lambda client: client.search_topics(keyword),
        render=render_topics,
        as_json=as_json,
        as_yaml=as_yaml,
    )


@click.command("sub-comments")
@click.argument("note_id")
@click.argument("comment_id")
@click.option("--cursor", default="", help="Pagination cursor")
@structured_output_options
@click.pass_context
def sub_comments(ctx, note_id: str, comment_id: str, cursor: str, as_json: bool, as_yaml: bool):
    """View replies to a specific comment."""
    handle_command(
        ctx,
        action=lambda client: client.get_sub_comments(note_id, comment_id, cursor=cursor),
        render=render_comments,
        as_json=as_json,
        as_yaml=as_yaml,
    )


@click.command("search-user")
@click.argument("keyword")
@structured_output_options
@click.pass_context
def search_user(ctx, keyword: str, as_json: bool, as_yaml: bool):
    """Search for users by keyword."""
    handle_command(
        ctx,
        action=lambda client: client.search_users(keyword),
        render=render_users,
        as_json=as_json,
        as_yaml=as_yaml,
    )


HOT_CATEGORIES = {
    "fashion": "homefeed.fashion_v3",
    "food": "homefeed.food_v3",
    "cosmetics": "homefeed.cosmetics_v3",
    "movie": "homefeed.movie_and_tv_v3",
    "career": "homefeed.career_v3",
    "love": "homefeed.love_v3",
    "home": "homefeed.household_product_v3",
    "gaming": "homefeed.gaming_v3",
    "travel": "homefeed.travel_v3",
    "fitness": "homefeed.fitness_v3",
}


@click.command()
@click.option(
    "--category", "-c",
    type=click.Choice(list(HOT_CATEGORIES.keys())),
    default="food",
    help="Category (fashion, food, cosmetics, movie, career, love, home, gaming, travel, fitness)",
)
@structured_output_options
@click.pass_context
def hot(ctx, category: str, as_json: bool, as_yaml: bool):
    """Browse hot/trending notes by category."""
    def _hot_action(client):
        result = client.get_hot_feed(HOT_CATEGORIES[category])
        _cache_tokens_from_items(result, xsec_source="pc_feed")
        save_index_from_items(result, xsec_source="pc_feed")
        return result

    handle_command(
        ctx,
        action=_hot_action,
        render=render_feed,
        as_json=as_json,
        as_yaml=as_yaml,
    )
