-- FAWP Scanner — Plan gating migration
-- Run in Supabase SQL Editor after migrations.sql

-- 1. Profiles table (mirrors auth.users, adds plan)
create table if not exists profiles (
    id         uuid references auth.users(id) on delete cascade primary key,
    email      text,
    plan       text not null default 'free',  -- free | pro | admin
    created_at timestamptz default now()
);

alter table profiles enable row level security;

create policy "Users read own profile"
    on profiles for select
    using (auth.uid() = id);

create policy "Users update own profile"
    on profiles for update
    using (auth.uid() = id)
    with check (plan = (select plan from profiles where id = auth.uid()));

-- Admin can update any profile (for manual plan upgrades)
create policy "Admin update any profile"
    on profiles for update
    using (
        (select plan from profiles where id = auth.uid()) = 'admin'
    );

create policy "Admin read all profiles"
    on profiles for select
    using (
        (select plan from profiles where id = auth.uid()) = 'admin'
    );

-- 2. Auto-insert profile on signup
create or replace function public.handle_new_user()
returns trigger language plpgsql security definer set search_path = public
as $$
begin
    insert into public.profiles (id, email, plan)
    values (new.id, new.email, 'free')
    on conflict (id) do nothing;
    return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
    after insert on auth.users
    for each row execute procedure public.handle_new_user();


-- 3. Plan limits reference (informational — enforced in app)
-- free:  3 tickers, 10 history snapshots, no alerts, no schedule
-- pro:   unlimited tickers, 500 history, full alerts, schedule
-- admin: everything + admin panel
