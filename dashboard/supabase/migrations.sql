-- FAWP Scanner — Supabase table migrations
-- Run this in Supabase SQL Editor (Dashboard → SQL Editor → New query)

-- 1. Scan history (per-user snapshots)
create table if not exists fawp_scan_history (
    id          bigserial primary key,
    user_id     uuid references auth.users(id) on delete cascade not null,
    scanned_at  timestamptz default now() not null,
    label       text default '',
    n_assets    int default 0,
    n_flagged   int default 0,
    payload     jsonb not null default '[]'
);

-- Index for fast per-user queries
create index if not exists fawp_scan_history_user_idx
    on fawp_scan_history(user_id, scanned_at desc);

-- RLS: users can only see their own scans
alter table fawp_scan_history enable row level security;

create policy "Users read own scans"
    on fawp_scan_history for select
    using (auth.uid() = user_id);

create policy "Users insert own scans"
    on fawp_scan_history for insert
    with check (auth.uid() = user_id);

create policy "Users delete own scans"
    on fawp_scan_history for delete
    using (auth.uid() = user_id);


-- 2. Watchlists (per-user named lists)
create table if not exists fawp_watchlists (
    id           bigserial primary key,
    user_id      uuid references auth.users(id) on delete cascade not null,
    name         text not null,
    config       jsonb not null default '{}',
    created_at   timestamptz default now() not null,
    last_scanned timestamptz,
    unique(user_id, name)
);

create index if not exists fawp_watchlists_user_idx
    on fawp_watchlists(user_id);

alter table fawp_watchlists enable row level security;

create policy "Users read own watchlists"
    on fawp_watchlists for select
    using (auth.uid() = user_id);

create policy "Users insert own watchlists"
    on fawp_watchlists for insert
    with check (auth.uid() = user_id);

create policy "Users update own watchlists"
    on fawp_watchlists for update
    using (auth.uid() = user_id);

create policy "Users delete own watchlists"
    on fawp_watchlists for delete
    using (auth.uid() = user_id);


-- 3. Scheduled scans
create table if not exists fawp_schedules (
    id           bigserial primary key,
    user_id      uuid references auth.users(id) on delete cascade not null,
    watchlist    text not null,
    cron_expr    text not null default '0 9 * * 1-5',
    email_alerts boolean default true,
    active       boolean default true,
    last_run     timestamptz,
    created_at   timestamptz default now() not null
);

alter table fawp_schedules enable row level security;

create policy "Users manage own schedules"
    on fawp_schedules for all
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);
