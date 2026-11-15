import { useEffect, useState } from "react";
import { Pencil, Plus, Trash2 } from "lucide-react";
import { Link } from "@tanstack/react-router";
import { Button } from "./ui/button";
import { supabase } from "@/integrations/supabase/client";
import { clearMemoryCache, readMemories, saveMemory, deleteMemory, type CharacterMemory } from "@/lib/memories";

export function MemoryPanel({ characterId, name }: { characterId: string; name: string }) {
  const [signedIn, setSignedIn] = useState(false);
  const [rows, setRows] = useState<CharacterMemory[]>([]);
  const [content, setContent] = useState("");
  const [editing, setEditing] = useState<string>();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let cancelled = false;
    async function refresh() {
      const { data: { session } } = await supabase.auth.getSession();
      if (cancelled) return;
      setSignedIn(!!session);
      if (!session) { clearMemoryCache(); setRows([]); return; }
      try { const memories = await readMemories(characterId); if (!cancelled) setRows(memories); }
      catch (e) { if (!cancelled) setError(e instanceof Error ? e.message : "No se pudieron cargar los recuerdos."); }
    }
    void refresh();
    const { data: { subscription } } = supabase.auth.onAuthStateChange(() => { void refresh(); });
    return () => { cancelled = true; subscription.unsubscribe(); };
  }, [characterId]);
  return <section className="memory-panel" aria-label="Memoria del personaje">
    <h2 className="text-lg font-semibold">Memoria de {name}</h2>
    {!signedIn ? <Button asChild variant="outline" className="mt-4"><Link to="/account">Iniciar sesión para guardar recuerdos</Link></Button> : <>
      <form className="mt-4 flex items-start gap-2" onSubmit={async e => {
        e.preventDefault(); setBusy(true); setError("");
        try { setRows(await saveMemory(characterId, content, editing)); setContent(""); setEditing(undefined); }
        catch(e) { setError(e instanceof Error ? e.message : "No se pudo guardar."); }
        finally { setBusy(false); }
      }}>
        <textarea aria-label="Recuerdo" className="memory-input" rows={2} maxLength={2000} value={content} onChange={e => setContent(e.target.value)} placeholder="Algo que quieres que recuerde sobre ti…" />
        <Button type="submit" disabled={busy || !content.trim()} aria-label={editing ? "Guardar cambios" : "Guardar recuerdo"}><Plus />Guardar</Button>
      </form>
      {editing && <Button variant="ghost" onClick={() => { setEditing(undefined); setContent(""); }}>Cancelar</Button>}
      <div className="memory-list">{rows.map(m => <div className="memory-row" key={m.id}>
        <p className="min-w-0 flex-1 break-words text-sm">{m.content}</p>
        <Button size="icon" variant="ghost" title="Editar recuerdo" aria-label="Editar recuerdo" onClick={() => { setEditing(m.id); setContent(m.content); }}><Pencil /></Button>
        <Button size="icon" variant="ghost" title="Eliminar recuerdo" aria-label="Eliminar recuerdo" onClick={async () => { try { setRows(await deleteMemory(characterId, m.id)); } catch(e) { setError(e instanceof Error ? e.message : "No se pudo borrar."); } }}><Trash2 /></Button>
      </div>)}</div>
    </>}
    {error && <p role="alert" className="mt-3 text-sm text-destructive">{error}</p>}
  </section>;
}