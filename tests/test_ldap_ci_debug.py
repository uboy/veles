"""Временный диагностический тест CI: разобрать, почему connect даёт
struct.error на ubuntu-раннере (локально не воспроизводится). Удалить после
диагностики."""
from __future__ import annotations

import json
import os
import socket

import pytest


def test_debug_ldap_candidates(ldap_server):
    import ldap3

    url = os.environ["VELES_LDAP_URL"]
    s = ldap3.Server(url, connect_timeout=5, get_info=None)
    cands = list(s.candidate_addresses())
    print("CANDIDATES:", [(c[0], c[1], c[2], c[4]) for c in cands])
    for a in cands:
        try:
            sk = socket.socket(*a[:3])
            sk.settimeout(5)
            sk.connect(a[4])
            sk.close()
            print("CONNECT-OK:", a[4])
        except Exception as e:
            print("CONNECT-FAIL:", a[4], type(e).__name__, e)
    c = ldap3.Connection(s, user="cn=nobody", password="x", auto_bind=False)
    try:
        c.open()
        print("OPEN-OK")
    except Exception as e:
        print("OPEN-FAIL:", type(e).__name__, e)
    print("GROUP_MAP:", os.environ.get("VELES_LDAP_GROUP_MAP", "")[:80])
    print("URL:", url)
    assert False, "debug-test: см. вывод выше"
