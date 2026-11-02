# 06 · Inventario de worktrees y qué subir (2026-11-02)

Estado leído con `git status`/`git fetch` (solo lectura). **Nada empujado.** Los remotes `fbientrigo/*` son push-elegibles solo con orden explícita; `ShipSoft/FairShip` nunca (AGENT_POLICY §3).

## Hallazgo principal

Todo lo aprendido desde 2026-07-19 vive **solo en local**: `origin/main` de `ship_adaptive_muon_bg` está en `4c2abe8` (2026-07-19). Hay **44 commits** sin remote repartidos en 7 ramas locales (6 sin upstream). Un disco roto = se pierde la tesis.

## A. Repo `fbientrigo/ship_adaptive_muon_bg` (push-elegible)

| Rama / worktree | Commits sin remote | Contenido | Valor como evidencia | Propuesta |
|---|---|---|---|---|
| `main` (canónico es `feat/minimal-loop-v0`) | 9 | checkpoint de notas, queue, tilt | notas de cola/tilt | subir |
| `feat/minimal-loop-v0` (`ship_adaptive_muon_bg/`, dirty 4: `notes/branches.md`, `notes/agents.md`, 2 `.txt` de sesión) | 19 | iteraciones 1-4 del lazo mínimo, seed repetido | **alta**: filtro ruidoso, presupuesto DIS barato | subir (los archivos dirty son de otra sesión, no se tocan) |
| `feat/p2-extraction-v0` (este worktree) | 16 | extracción P2 sin pérdida, procedencia, driver, `green_field_notes/` | **alta**: base del greenfield | subir |
| `notes/branch-triage-v0` (`ship-notes-triage`, limpio) | 15 (6 propios) | origen de +2.5916 m, rendimiento stage-1, generación desde cero | **alta** (aclara el offset heredado) | subir |
| `fix/issue-45-extract-v0` | 15 (6 propios) | fixes de procedencia en evaluación/pilot/batch | media: correcciones con tests | subir |
| `chore/sbt-single-predicate-v0` | 10 (1 propio) | un solo predicado SBT | media | subir |
| `demo/pi-utility-guided-v0` (`ship-pi-demo-v0`, limpio, 1.3 GB en disco) | 5 | demo π guiado por utilidad | media | subir rama; los 1.3 GB son datos locales, no van |
| `tmp-merge-pi-demo` (**merge a medias**, 69 archivos staged, HEAD desacoplado `f74c028`) | – | resolución de conflictos de un merge | baja: temporal | **no tocar**; pregunta abajo |
| 37 ramas solo-remotas (`experiment/d9-*`, `sprint1/*`, `claude/*`…) | – | campaña histórica D5-D9 | histórica | candidatas a borrar en remoto (acción remota: solo con tu orden) |

Union de lo no subido: 44 commits. Archivo tracked más grande 1.6 MB (muestras `data/samples/*`), nada >2 MB; `.git` 103 MB.

## B. Otros repos `fbientrigo/*`

| Repo | Estado | Propuesta |
|---|---|---|
| `NFlow` | limpio, 0 ahead | nada que subir |
| `phd-thesis` | limpio, 0 ahead | nada que subir |
| `root-rag` | 42 archivos sin commit (21 M, 21 ??), 0 ahead, 5.8 GB en disco | revisar aparte; no es evidencia de esta tesis; preguntar |

## C. FairShip (`ShipSoft/FairShip`, remoto de solo lectura)

No se commitea ni se empuja. Lo útil se exporta como parche al repo de tesis (`evidence/fairship_local_patches/`, base en `BASES.txt`).

| Worktree | Sucio | Veredicto | Acción |
|---|---|---|---|
| `FairShip-mudis-compat` (`investigation/mudis-root-compat-0808`) | `makeMuonDIS.py` (+16/-10), ~14 `.root` de salida | **evidencia útil** (compatibilidad ROOT, documentada en `docs/0808_*`) | parche exportado; `.root` quedan locales |
| `FairShip-pr1316-positive-ab` (HEAD desacoplado) | 4 `.py` de `muonDIS/` (+11/-11) | evidencia útil (acceso por `file["name"]`, PR #1316) | parche exportado |
| `FairShip-issue1524` (`fix/calo-pid-duplicate-yaml-keys`) | 2 archivos (+152/-1) con tests, 3 `codex-session-*.md` (~840 kB) | fix real para humano; las sesiones son ruido | parche exportado; sesiones: candidatas a archivar fuera |
| `FairShip-AM-muDIS` (`thesis/am-mudis`) | `CLAUDE.md`, salidas `.root` | evidencia de bridge-1281 ya documentada | nada más |
| `FairShip-pr1432-400c0fb` | `.pixi`, un `.txt` pegado (148 kB) | ruido | candidata a limpiar |
| `FairShip-main-8a1cc53`, `-8df35bc`, `-current-main-f73a305` | limpios | pins de la tesis | conservar (reconstruibles con `git worktree add` desde el pin) |
| `FairShip/` (canónico), `FairShip-pr1316` | limpios | – | – |

Cada worktree FairShip pesa ~7-8 GB (los 8 suman ~60 GB). Los pins son reconstruibles desde un solo clon; mantener 1 clon + 3 pins reduce el disco a ~1/3. Decisión tuya.

## D. Ruido (no subir)

- `*-this-session-is-being-continued-*.txt`, `*-pastedcontent-*.txt` (exports de sesión, 150-200 kB cada uno) en 4 sitios.
- `codex-session-*.md` en `FairShip-issue1524`.
- `scratch/` (2.0 GB): salidas ROOT/numpy; la evidencia que importa ya está resumida en `docs/` y `notes/`. Debe seguir local hasta decidir qué pasar a un archivo con manifiesto + sha256 (no se borra).

## E. Para el agente de factibilidad (greenfield)

Quien explore la factibilidad debería leer, en este orden: `03_lessons_learned.md`, `notes/p2_lossless_v0.md`, `notes/muon_generation_from_scratch.md` (rama `notes/branch-triage-v0`), `docs/decisions/stage_outcomes_v0.md`, la nota del lazo mínimo en `feat/minimal-loop-v0`. Hasta que esas ramas no estén en el remoto, solo existe en esta máquina.

## Preguntas abiertas para el push

1. ¿Empujo las 7 ramas del repo de tesis (44 commits, ~1 push por rama, sin force)? Lista exacta arriba.
2. `tmp-merge-pi-demo`: ¿lo aborto/descarto (aborta un merge no commiteado; irreversible para esa resolución) o lo dejas?
3. `root-rag`: ¿commit de los 42 archivos?
4. ¿Borrar en remoto las 37 ramas históricas? (acción remota).
