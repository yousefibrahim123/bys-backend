"""
AI assistant orchestration.

The loop: plan which tools to use -> run them against the database and the
models -> compose the reply from those results only.

With no LLM configured, the local engine runs the exact same loop — it picks
tools by keyword and composes the reply from templates. The numbers are
identical either way; only the phrasing differs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal, Any

from sqlalchemy.orm import Session

from ..config import settings

from . import llm
from .tools import TOOL_SPECS, run_tool

log = logging.getLogger("ai.assistant")

MAX_TOOLS = 3

PLANNER_SYSTEM = """You are a tool planner for a pharmacy management platform.
Your job: choose the tools needed to answer the user's question. Do not answer it.

Available tools:
{tools}

Return JSON only, in exactly this shape, with no other text:
{{"tools": [{{"name": "tool_name", "args": {{}}}}]}}

Rules:
- At most {max_tools} tools.
- For a general stock-health question use inventory_overview.
- For a question about one specific product use forecast_product with product_name.
- If no tool is needed return {{"tools": []}}."""

ANSWER_SYSTEM = """You are the AI assistant for BYS, a pharmacy supply-chain
platform. You are helping a pharmacist at the {branch} branch.

Strict rules:
- Use only the numbers in the attached data. Never invent a number or a product name.
- If the data doesn't answer the question, say so plainly and suggest another one.
- Always reply in English, whatever language the question was asked in.
- Be brief and practical: 2-5 sentences, then one clear recommendation.
- Use **bold** for product names and key figures.
- Currency is Egyptian pounds, written as EGP.
- When quoting a forecast say it came from the forecast model; when quoting an
  expiry risk say it came from the expiry model.
- Earlier turns are context only. Answer the newest question on its own — never
  restate, continue, or append to an answer you already gave. If the new
  question repeats an earlier one, answer it again fresh and concisely rather
  than adding to the previous reply."""


# ------------------------------------------------------------ tool selection

KEYWORDS: List[Tuple[Tuple[str, ...], str, dict]] = [
    (("run out", "runout", "run low", "stock out", "stockout", "out of stock",
      "low stock", "low on stock", "running low", "understock", "below minimum",
      "below the minimum", "reorder point", "shortage", "short on",
      "need restocking", "restock", "replenish", "\u064a\u0646\u0641\u062f",
      "\u0647\u064a\u0646\u0641\u062f", "\u0647\u062a\u062e\u0644\u0635",
      "\u0628\u062a\u062e\u0644\u0635", "\u064a\u062e\u0644\u0635",
      "\u062e\u0644\u0635\u0627\u0646", "\u0639\u0644\u0649 \u0648\u0634\u0643",
      "\u0646\u0627\u0642\u0635", "\u0646\u0627\u0641\u062f",
      "\u0627\u0644\u062d\u062f \u0627\u0644\u0627\u062f\u0646\u0649",
      "\u0642\u0644\u064a\u0644"), "low_stock", {}),
    (("expiry", "expire", "expires", "expiring", "expired", "shelf life",
      "near expiry", "close to expiry", "about to expire", "wastage", "waste",
      "\u0635\u0644\u0627\u062d\u064a\u0647", "\u062a\u0646\u062a\u0647\u064a",
      "\u0645\u0646\u062a\u0647\u064a", "\u0642\u0631\u0628"), "near_expiry", {}),
    (("profit", "profits", "margin", "margins", "profitable", "profitability",
      "best seller", "bestseller", "top product", "top products", "earning",
      "\u0631\u0628\u062d", "\u0631\u0628\u062d\u064a\u0647",
      "\u0647\u0627\u0645\u0634", "\u0627\u064a\u0631\u0627\u062f",
      "\u0645\u0643\u0633\u0628"), "top_products", {"metric": "profit"}),
    # Existing purchase orders and their statuses \u2014 checked before the reorder
    # suggestions so "my orders" doesn't turn into a buying list.
    (("my orders", "our orders", "order status", "orders status",
      "purchase orders", "pending orders", "incoming orders", "open orders",
      "track order", "order tracking", "\u0627\u0648\u0631\u062f\u0631", "\u0627\u0648\u0631\u062f\u0631\u0627\u062a", "\u0627\u0644\u0627\u0648\u0631\u062f\u0631",
      "\u0637\u0644\u0628\u0627\u062a\u064a", "\u0627\u0644\u0637\u0644\u0628\u0627\u062a", "\u062d\u0627\u0644\u0647 \u0627\u0644\u0637\u0644\u0628", "\u0645\u062a\u0627\u0628\u0639\u0647 \u0627\u0644\u0637\u0644\u0628"), "orders_summary", {}),
    (("transfer", "transfers", "stock transfer", "between branches",
      "\u062a\u062d\u0648\u064a\u0644", "\u0627\u0644\u062a\u062d\u0648\u064a\u0644\u0627\u062a", "\u0646\u0642\u0644 \u0645\u062e\u0632\u0648\u0646", "\u0646\u0642\u0644 \u0628\u064a\u0646"), "transfers_summary", {}),
    (("customer", "customers", "client", "clients", "\u0639\u0645\u064a\u0644", "\u0639\u0645\u0644\u0627\u0621",
      "\u0627\u0644\u0632\u0628\u0627\u064a\u0646", "\u0632\u0628\u0648\u0646"), "customers_summary", {}),
    (("offer", "offers", "discount", "discounts", "deal", "deals", "promotion",
      "catalogue", "catalog", "listings", "\u0639\u0631\u0636", "\u0639\u0631\u0648\u0636", "\u062e\u0635\u0645", "\u062e\u0635\u0648\u0645\u0627\u062a",
      "\u062a\u062e\u0641\u064a\u0636"), "supplier_offers", {}),
    (("reorder", "re-order", "order list", "purchase list",
      "what should i order", "what to order", "buy list", "procurement",
      "\u0627\u0637\u0644\u0628", "\u0637\u0644\u0628\u064a\u0647",
      "\u0627\u0639\u0627\u062f\u0647 \u0637\u0644\u0628",
      "\u0627\u0639\u0627\u062f\u0647 \u0627\u0644\u0637\u0644\u0628",
      "\u0642\u0627\u0626\u0645\u0647 \u0637\u0644\u0628",
      "\u0634\u0631\u0627\u0621", "\u0627\u0634\u062a\u0631\u064a",
      "\u0646\u0637\u0644\u0628"), "reorder_list", {}),
    (("forecast", "forecasting", "predict", "prediction", "demand", "projected",
      "next week", "next month", "how many will", "\u062a\u0648\u0642\u0639",
      "\u062a\u0646\u0628\u0624", "\u0627\u0644\u0637\u0644\u0628"),
     "forecast_product", {}),
    (("sales", "sold", "selling", "revenue", "turnover", "takings",
      "\u0645\u0628\u064a\u0639\u0627\u062a", "\u0628\u0639\u062a"),
     "sales_summary", {}),
    (("supplier", "suppliers", "vendor", "vendors", "distributor",
      "\u0645\u0648\u0631\u062f", "\u0645\u0648\u0631\u062f\u064a\u0646"),
     "supplier_performance", {}),
    (("summary", "summarize", "summarise", "overview", "health", "inventory",
      "stock level", "how are we doing", "status", "\u0645\u0644\u062e\u0635",
      "\u0644\u062e\u0635", "\u0646\u0638\u0631\u0647",
      "\u062d\u0627\u0644\u0647", "\u0627\u0644\u0648\u0636\u0639",
      "\u0627\u0644\u0645\u062e\u0632\u0648\u0646",
      "\u0645\u062e\u0632\u0648\u0646"), "inventory_overview", {}),
]


def _norm_ar(t: str) -> str:
    """Normalise Arabic letter variants so keyword matching survives spelling
    differences. Input-side only — replies are always English."""
    for a, b in (("\u0623", "\u0627"), ("\u0625", "\u0627"), ("\u0622", "\u0627"),
                 ("\u0649", "\u064a"), ("\u0629", "\u0647")):
        t = t.replace(a, b)
    return t


# Filler words dropped before matching, so "low ON stock" and "run OUT OF
# stock" reach the same keyword as "low stock" / "run out stock".
_FILLER = {"the", "a", "an", "of", "on", "in", "at", "to", "for", "my", "our",
           "is", "are", "was", "were", "do", "does", "did", "we", "i",
           "please", "can", "could", "would", "should", "any", "some"}


def _skeleton(text: str) -> str:
    """
    Lowercased, Arabic-normalised, filler-stripped, single-spaced.

    Matching used to be a raw substring test, so a keyword only fired on an
    exact phrasing. "Which products are low on stock?" missed every keyword
    because the table had "low stock", not "low on stock" — so the planner
    fell through to its default and answered with the same generic inventory
    summary for many different questions. That is what made the assistant look
    like it kept repeating itself.
    """
    cleaned = _norm_ar(text.lower())
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in cleaned)
    words = [w for w in cleaned.split() if w and w not in _FILLER]
    return " " + " ".join(words) + " "


def _matches(keys: Tuple[str, ...], skel: str, raw: str) -> bool:
    for k in keys:
        kn = _norm_ar(k.lower())
        if kn in raw:                      # exact phrase, as before
            return True
        ks = _skeleton(k).strip()
        if ks and f" {ks} " in skel:       # filler-insensitive phrase
            return True
    return False


def plan_locally(message: str) -> List[dict]:
    raw = _norm_ar(message.lower())
    skel = _skeleton(message)
    picked: List[dict] = []
    for keys, tool, args in KEYWORDS:
        if _matches(keys, skel, raw) and not any(p["name"] == tool for p in picked):
            picked.append({"name": tool, "args": dict(args)})
        if len(picked) >= MAX_TOOLS:
            break
    # Check for simple greetings or small talk
    greetings = {"hi", "hello", "hey", "say hi", "say hello", "say any think alse", "say anything else", "hi there", "salam", "marhaba", "ahlan", "ezayak", "azayak", "good morning", "good evening", "good afternoon"}
    msg_clean = skel.strip().lower()
    if msg_clean in greetings or raw.strip() in {"hi", "hello", "hey", "say hi", "salam", "ahlan", "مرحبا", "ازيك", "ازيكوا", "سلام"}:
        return []

    if not picked:
        picked = [{"name": "inventory_overview", "args": {}}]

    # forecast_product needs a product name — fall back to the reorder list
    for p in picked:
        if p["name"] == "forecast_product" and not p["args"].get("product_name"):
            guess = _guess_product_name(message)
            if guess:
                p["args"]["product_name"] = guess
            else:
                p["name"] = "reorder_list"
                p["args"] = {}
    return picked[:MAX_TOOLS]


def _guess_product_name(message: str) -> str:
    """Longest word that isn't a common question word — usually the product name."""
    stop = {"forecast", "predict", "demand", "for", "the", "what", "which", "how",
            "many", "much", "next", "week", "month", "days", "product", "توقع",
            "تنبؤ", "الطلب", "على", "كام", "ايه", "إيه", "في", "من", "لل"}
    words = [w.strip(".,؟?!") for w in message.split()]
    cands = [w for w in words if len(w) > 3 and w.lower() not in stop]
    return max(cands, key=len) if cands else ""


async def plan_with_llm(message: str) -> Tuple[List[dict], bool]:
    tools_desc = "\n".join(
        f"- {t['name']}: {t['description']}"
        + (f" | args: {t['args']}" if t["args"] else "")
        for t in TOOL_SPECS)
    system = PLANNER_SYSTEM.format(tools=tools_desc, max_tools=MAX_TOOLS)
    res = await llm.complete(system, [{"role": "user", "content": message}], json_mode=True)
    if not res.ok:
        return plan_locally(message), False
    data = llm.extract_json(res.text)
    if not isinstance(data, dict) or not isinstance(data.get("tools"), list):
        return plan_locally(message), False
    plan = []
    for t in data["tools"][:MAX_TOOLS]:
        if isinstance(t, dict) and isinstance(t.get("name"), str):
            plan.append({"name": t["name"],
                         "args": t.get("args") if isinstance(t.get("args"), dict) else {}})
    return (plan or plan_locally(message)), True


# ------------------------------------------------------- local reply composer

def _fmt(n: float) -> str:
    return f"{n:,.0f}"


def answer_locally(message: str, branch: str, results: Dict[str, dict]) -> str:
    """
    Compose the reply from tool results only.

    Replies are English-only. Arabic keywords are still recognised on the way
    in (see # Input-side only: these Arabic keywords let an Arabic question still select
# the right tool. Nothing here is ever shown to the user — replies are
# always composed in English.
KEYWORD_TOOLS) so an Arabic question still selects the right
    tools — only the output language is fixed.
    """
    parts: List[str] = []

    if (r := results.get("inventory_overview")) and "error" not in r:
        parts.append(
            f"**{branch}** branch: **{r['total_products']}** products, "
            f"**{_fmt(r['total_units'])}** units, stock value "
            f"**EGP {_fmt(r['inventory_value_egp'])}**, potential profit "
            f"**EGP {_fmt(r['potential_profit_egp'])}**. "
            f"**{r['low_stock_count']}** below minimum, "
            f"**{r['out_of_stock_count']}** out of stock, "
            f"**{r['near_expiry_count']}** nearing expiry.")

    if (r := results.get("low_stock")) and "error" not in r:
        if r["count"] == 0:
            parts.append("All products are currently above their minimum levels.")
        else:
            top = r["items"][:3]
            parts.append(
                f"**{r['count']}** products are below minimum stock. Most urgent: " +
                ", ".join(f"**{i['name']}** ({i['current_stock']} units, "
                          f"{i['days_until_empty']:.0f} days left)" for i in top) + ".")

    if (r := results.get("near_expiry")) and "error" not in r:
        if r["count"] == 0:
            parts.append("No products are nearing expiry.")
        else:
            worst = r["items"][0]
            parts.append(
                f"**{r['count']}** products near expiry with **EGP "
                f"{_fmt(r['total_value_at_risk_egp'])}** at risk. Highest: "
                f"**{worst['name']}** ({worst['quantity']} units, "
                f"{worst['days_to_expiry']} days left, {worst['risk_probability']:.0%} "
                f"risk) — recommended: **{worst['action']}**. [expiry model]")

    if (r := results.get("top_products")) and r.get("items"):
        t = r["items"][0]
        parts.append(
            f"Top product by profit: **{t['name']}** — monthly profit "
            f"**EGP {_fmt(t['monthly_profit_egp'])}** at a **{t['margin_pct']:.0f}%** margin.")

    if (r := results.get("forecast_product")):
        if "error" in r:
            parts.append(r["error"])
        else:
            parts.append(
                f"28-day forecast for **{r['product']}**: "
                f"**{r['forecast_28d_p50']:.0f}** units (likely), "
                f"**{r['forecast_28d_p90']:.0f}** (safe). Stock of "
                f"**{r['current_stock']}** covers **{r['days_of_cover']:.0f}** days. "
                + (f"Order **{r['recommended_order_qty']}** units."
                   if r["should_order_now"] else "No reorder needed right now.")
                + " [forecast model]")

    if (r := results.get("reorder_list")) and "error" not in r:
        if r["count"] == 0:
            parts.append("Nothing needs reordering right now.")
        else:
            top = r["items"][:3]
            parts.append(
                f"**{r['count']}** products need reordering, estimated "
                f"**EGP {_fmt(r['total_estimated_cost_egp'])}**. Priority: "
                + ", ".join(f"**{i['name']}** ({i['recommended_qty']} units)" for i in top)
                + ". [forecast model]")

    if (r := results.get("sales_summary")) and "error" not in r:
        parts.append(
            f"Last **{r['days']}** days: revenue **EGP {_fmt(r['total_revenue_egp'])}** "
            f"across **{r['transactions']}** transactions, average basket "
            f"**EGP {r['average_basket_egp']:.0f}**.")

    if (r := results.get("supplier_performance")) and r.get("suppliers"):
        s = r["suppliers"][0]
        parts.append(
            f"Top supplier **{s['supplier']}**: **{s['orders']}** orders worth "
            f"**EGP {_fmt(s['total_egp'])}**, delivery rate "
            f"**{s['delivery_rate_pct']:.0f}%**.")

    if (r := results.get("search_product")) and "error" not in r:
        if not r.get("items"):
            parts.append("No matching products found.")
        else:
            parts.append(", ".join(
                f"**{i['name']}** ({i['quantity']} x EGP {i['selling_price']:.0f})"
                for i in r["items"][:5]))

    if (r := results.get("orders_summary")) and "error" not in r:
        if r["total_orders"] == 0:
            parts.append("No purchase orders yet — create one from Purchase Orders.")
        else:
            st = ", ".join(f"{v} {k}" for k, v in r["by_status"].items())
            latest = r["recent"][0]
            parts.append(
                f"**{r['total_orders']}** purchase orders ({st}), total "
                f"**EGP {_fmt(r['total_value_egp'])}** with "
                f"**EGP {_fmt(r['open_value_egp'])}** still open. Latest: "
                f"**{latest['order_number']}** to **{latest['supplier']}** — "
                f"{latest['status']}, EGP {_fmt(latest['total_egp'])}.")

    if (r := results.get("transfers_summary")) and "error" not in r:
        if r["total_transfers"] == 0:
            parts.append("No stock transfers involve this branch yet.")
        else:
            st = ", ".join(f"{v} {k}" for k, v in r["by_status"].items())
            parts.append(f"**{r['total_transfers']}** transfer requests ({st}).")

    if (r := results.get("customers_summary")) and "error" not in r:
        if r["total_customers"] == 0 and not r["top_customers"]:
            parts.append("No customers are registered for this branch yet.")
        else:
            line = f"**{r['total_customers']}** registered customers."
            if r["top_customers"]:
                t = r["top_customers"][0]
                line += (f" Top spender: **{t['name']}** — EGP "
                         f"{_fmt(t['total_spent_egp'])} over {t['purchases']} purchases.")
            parts.append(line)

    if (r := results.get("supplier_offers")) and "error" not in r:
        if not r.get("items"):
            parts.append("No supplier listings match that right now.")
        else:
            deals = [i for i in r["items"] if i["offer_percent"] > 0]
            shown = (deals or r["items"])[:3]
            parts.append(
                "Supplier catalogue: " + ", ".join(
                    f"**{i['product']}** from **{i['supplier']}** at "
                    f"EGP {i['price_after_offer_egp']:.0f}"
                    + (f" (−{i['offer_percent']:.0f}%)" if i["offer_percent"] else "")
                    for i in shown) + ".")

    if not parts:
        raw_msg = message.lower().strip()
        if any(g in raw_msg for g in ["hi", "hello", "hey", "salam", "ahlan", "marhaba", "مرحبا", "ازيك", "سلام"]):
            return f"Hello! How can I help you today with inventory, sales, suppliers, or forecasting for the **{branch}** branch?"
        return ("I couldn't pull data for that. Try asking about inventory health, "
                "low stock, near-expiry items, your purchase orders, transfers, "
                "customers, supplier offers, or the reorder list.")
    return " ".join(parts)


# ------------------------------------------------------------------- charts

def build_chart(results: Dict[str, dict]) -> Optional[dict]:
    if (r := results.get("low_stock")) and r.get("items"):
        return {"label": "Current stock (units)",
                "data": [{"name": i["name"].split()[0], "value": i["current_stock"]}
                         for i in r["items"][:5]]}
    if (r := results.get("near_expiry")) and r.get("items"):
        return {"label": "Expected loss (EGP)",
                "data": [{"name": i["name"].split()[0], "value": i["expected_loss_egp"]}
                         for i in r["items"][:5]]}
    if (r := results.get("top_products")) and r.get("items"):
        return {"label": "Monthly profit (EGP)",
                "data": [{"name": i["name"].split()[0], "value": i["monthly_profit_egp"]}
                         for i in r["items"][:5]]}
    if (r := results.get("reorder_list")) and r.get("items"):
        return {"label": "Recommended qty (units)",
                "data": [{"name": i["name"].split()[0], "value": i["recommended_qty"]}
                         for i in r["items"][:5]]}
    if (r := results.get("sales_summary")) and r.get("top_products"):
        return {"label": "Revenue (EGP)",
                "data": [{"name": t["name"].split()[0], "value": t["revenue_egp"]}
                         for t in r["top_products"][:5]]}
    if (r := results.get("forecast_product")) and "error" not in (r or {}):
        return {"label": "Forecast vs stock (units)",
                "data": [{"name": "Stock", "value": r["current_stock"]},
                         {"name": "Forecast P50", "value": round(r["forecast_28d_p50"])},
                         {"name": "Forecast P90", "value": round(r["forecast_28d_p90"])},
                         {"name": "Recommended order", "value": r["recommended_order_qty"]}]}
    return None


# --------------------------------------------------------------- main loop

async def answer(db: Session, message: str, branch: str,
                 history: Optional[List[dict]] = None,
                 conversation_id: Optional[str] = None) -> dict:
    """
    One turn.

    Latency matters here. Previously every turn made *two* LLM calls — one to
    plan tools, one to write the answer — each allowed up to the full Ollama
    timeout. On a large local model that is minutes per reply with no output,
    which is exactly what made the assistant appear to hang forever.

    Now: the keyword planner runs locally (instant, and it routes correctly),
    and only the answer goes to the LLM under a hard time budget. If the
    budget expires we return the locally composed answer, so a reply always
    comes back.
    """
    conv = conversation_id or uuid.uuid4().hex[:16]
    started = time.monotonic()

    if settings.ai_llm_planner:
        plan, _ = await plan_with_llm(message)
    else:
        plan = plan_locally(message)

    results: Dict[str, dict] = {}
    for step in plan:
        results[step["name"]] = run_tool(step["name"], db, branch, step.get("args"))

    chart = build_chart(results)
    local_reply = answer_locally(message, branch, results)
    reply, provider, degraded = local_reply, "local", False

    remaining = settings.ai_reply_budget_seconds - (time.monotonic() - started)
    if remaining > 5:
        context = json.dumps(results, ensure_ascii=False, default=str)[:12000]
        # Trim to whole turns. Slicing a flat list could start on an assistant
        # message, which invites the model to keep extending that answer
        # instead of replying to the new question.
        prior = list(history or [])
        while prior and prior[0]["role"] != "user":
            prior.pop(0)
        msgs = prior[-6:]
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)
        while msgs and msgs[-1]["role"] == "user":
            msgs.pop()
        msgs.append({"role": "user",
                     "content": f"User question: {message}\n\n"
                                f"Real system data (use only this):\n{context}"})
        try:
            res = await asyncio.wait_for(
                llm.complete(ANSWER_SYSTEM.format(branch=branch), msgs),
                timeout=remaining)
            if res.ok and res.text:
                reply, provider = res.text, res.provider
        except asyncio.TimeoutError:
            # The model is too slow for interactive use. The user still gets a
            # complete, correct answer built from the same tool results.
            degraded = True
            log.info("LLM exceeded the %.0fs budget — served the local answer",
                     settings.ai_reply_budget_seconds)

    return {
        "conversation_id": conv,
        "reply": reply,
        "chart": chart,
        "tools_used": list(results.keys()),
        "provider": provider,
        "grounded": True,
        "degraded": degraded,
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }
