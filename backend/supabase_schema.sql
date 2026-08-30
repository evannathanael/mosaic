-- Run this once in Supabase Dashboard -> SQL Editor.
-- Create a public Storage bucket named mosaic-images in Storage first.

create table if not exists public.posts (
    image_id text primary key,
    account_id text not null,
    handle text not null,
    thumbnail_url text not null,
    ai_probability double precision not null check (ai_probability between 0 and 1),
    confidence double precision not null check (confidence between 0 and 1),
    robustness jsonb not null default '{}'::jsonb,
    similarity_cluster integer not null,
    repetition_score double precision not null check (repetition_score between 0 and 1),
    diversity_label text not null,
    created_at timestamptz not null,
    analysis_mode text not null default 'mock',
    source text not null default 'upload',
    image_sha256 text,
    width integer,
    height integer,
    image_format text,
    image_storage_key text,
    thumbnail_storage_key text,
    is_seed boolean not null default false
);

create index if not exists posts_created_at_idx on public.posts (created_at desc);
create index if not exists posts_cluster_idx on public.posts (similarity_cluster);
create index if not exists posts_hash_idx on public.posts (image_sha256);
create index if not exists posts_account_idx on public.posts (account_id);

alter table public.posts enable row level security;
