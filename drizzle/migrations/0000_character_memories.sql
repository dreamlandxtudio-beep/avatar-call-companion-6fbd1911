CREATE TABLE public.character_memories (
 id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
 user_id uuid NOT NULL DEFAULT auth.uid(),
 character_id text NOT NULL,
 content text NOT NULL CHECK (char_length(content) BETWEEN 1 AND 2000),
 created_at timestamptz NOT NULL DEFAULT now()
);
GRANT SELECT, INSERT, UPDATE, DELETE ON public.character_memories TO authenticated;
GRANT ALL ON public.character_memories TO service_role;
ALTER TABLE public.character_memories ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Own memories" ON public.character_memories FOR ALL TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
CREATE INDEX character_memories_owner_character ON public.character_memories(user_id, character_id);