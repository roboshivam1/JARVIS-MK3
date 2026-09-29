// ===========================================================================
// people-directory.js - the Directory half of /app/people
//
// Loaded after the page's own script, so it shares its helpers: api(),
// esc(), state, CATEGORY_NAMES, CLOSENESS. Split out because the page
// was getting long, and these two halves never talk to each other
// except through state.
// ===========================================================================

const GROUP_KINDS = ['family', 'school', 'college', 'locality', 'project', 'work', 'other'];
const KIND_NAMES = {
  family: 'Family', school: 'School', college: 'College', locality: 'Where you live',
  project: 'Projects', work: 'Work', other: 'Other',
};

function switchView(name) {
  view.current = name;
  $('review-view').hidden = name !== 'review';
  $('people-view').hidden = name !== 'people';
  for (const tab of document.querySelectorAll('.tab')) {
    tab.classList.toggle('on', tab.dataset.view === name);
  }
  if (name === 'people' && !state.dir) loadDirectory();
}

document.querySelector('.tabs').addEventListener('click', (event) => {
  const tab = event.target.closest('.tab');
  if (tab) switchView(tab.dataset.view);
});

// -- the directory ----------------------------------------------------------

async function loadDirectory(keepProfile) {
  try {
    state.dir = await api('/people/directory');
  } catch (error) {
    $('directory').innerHTML = `<p class="error">${esc(error.message)}</p>`;
    return;
  }
  renderDirectory();
  if (keepProfile && state.person) openPerson(state.person.id);
}

function renderDirectory() {
  const term = $('dir-search').value.trim().toLowerCase();
  const { people, groups } = state.dir;
  const matches = (p) => !term
    || p.name.toLowerCase().includes(term)
    || (p.aliases || []).some((a) => a.toLowerCase().includes(term));

  const byId = Object.fromEntries(people.map((p) => [p.id, p]));
  const sections = [];
  for (const kind of GROUP_KINDS) {
    for (const group of groups.filter((g) => g.kind === kind)) {
      const members = group.members.map((id) => byId[id]).filter(Boolean).filter(matches);
      if (members.length || !term) sections.push([group.name, members, group.id]);
    }
  }
  const loose = people.filter((p) => !p.groups.length).filter(matches);
  if (loose.length) sections.push(['Not in a group', loose, null]);

  $('directory').innerHTML = sections.length
    ? sections.map(([name, members, id]) => `
        <div class="dir-group">
          <span>${esc(name)}</span>
          ${id ? `<button data-act="drop-group" data-id="${id}">remove</button>` : ''}
        </div>
        ${members.length
          ? members.map((p) => `
              <div class="dir-person${state.person && state.person.id === p.id ? ' on' : ''}"
                   data-act="open" data-id="${p.id}">
                ${esc(p.name)}${p.aliases.length ? `<span class="aka"> ${esc(p.aliases[0])}</span>` : ''}
              </div>`).join('')
          : '<div class="empty">no one yet</div>'}`).join('')
    : '<p class="empty">No one here yet. Debriefs you approve show up in this list.</p>';
}

$('dir-search').addEventListener('input', renderDirectory);

$('new-group-kind').innerHTML = GROUP_KINDS
  .map((k) => `<option value="${k}">${esc(KIND_NAMES[k])}</option>`).join('');

$('add-group').addEventListener('click', async () => {
  const name = $('new-group').value.trim();
  if (name.length < 2) return;
  try {
    await api('/people/groups', { name, kind: $('new-group-kind').value });
    $('new-group').value = '';
    $('dir-status').textContent = '';
    await loadDirectory(true);
  } catch (error) {
    $('dir-status').textContent = error.message;
  }
});

// -- one person -------------------------------------------------------------

async function openPerson(id) {
  try {
    state.person = await api(`/people/person/${id}`);
  } catch (error) {
    $('profile').innerHTML = `<p class="error">${esc(error.message)}</p>`;
    return;
  }
  state.editing = null;
  renderProfile();
  renderDirectory();
}

function factRow(f, historical) {
  const source = f.debrief_id && state.person.debriefs[f.debrief_id];
  const note = historical
    ? (f.superseded_by ? 'replaced' : 'no longer true')
    : (f.sensitivity === 'sensitive' ? 'sensitive' : '');
  if (state.editing === f.id) {
    return `<div class="fact" data-id="${f.id}">
      <div class="edit">
        <label for="e-cat">Kind</label>
        <select id="e-cat">${Object.entries(CATEGORY_NAMES).map(([v, label]) =>
          `<option value="${v}"${v === f.category ? ' selected' : ''}>${esc(label)}</option>`).join('')}</select>
        <label for="e-text">Fact</label>
        <textarea id="e-text" rows="2">${esc(f.text)}</textarea>
        <label for="e-sens">Sensitivity</label>
        <select id="e-sens">
          <option value="normal"${f.sensitivity === 'normal' ? ' selected' : ''}>Normal</option>
          <option value="sensitive"${f.sensitivity === 'sensitive' ? ' selected' : ''}>Sensitive</option>
        </select>
      </div>
      <div class="actions">
        <button class="small primary" data-act="save-fact" data-id="${f.id}">Save</button>
        <button class="small" data-act="cancel-fact">Cancel</button>
      </div></div>`;
  }
  return `<div class="fact" data-id="${f.id}">
    <div>
      <div class="kind">${esc(CATEGORY_NAMES[f.category] || f.category)}
        ${note ? `<span class="${historical ? 'kind' : 'flag'}">${esc(note)}</span>` : ''}</div>
      <div class="${historical ? 'gone' : ''}">${esc(f.text)}</div>
      ${source ? `<div class="provenance" data-act="source" data-id="${f.debrief_id}">
        from what you said on ${esc(new Date(source.created_ts).toLocaleDateString())}</div>` : ''}
      ${state.showing === f.debrief_id && source
        ? `<div class="said">${esc(source.raw_text)}</div>` : ''}
    </div>
    ${historical ? '<div></div>' : `<div class="actions">
      <button class="small" data-act="edit-fact" data-id="${f.id}">Edit</button>
      <button class="small" data-act="retire-fact" data-id="${f.id}">Retire</button>
    </div>`}</div>`;
}

function renderProfile() {
  const p = state.person;
  if (!p) return;
  const others = state.dir.people.filter((x) => x.id !== p.id);
  const inGroups = new Set(p.groups.map((g) => g.id));
  const joinable = state.dir.groups.filter((g) => !inGroups.has(g.id));

  $('profile').innerHTML = `
    <h3>${esc(p.name)}</h3>
    <div class="sub">${esc([p.relationship, CLOSENESS[p.closeness],
        p.status !== 'active' ? p.status.replace('_', ' ') : '',
        p.how_met && `met ${p.how_met}`].filter(Boolean).join(' · '))}</div>
    ${p.summary ? `<div>${esc(p.summary)}</div>` : ''}
    <div class="row-end">
      <button class="small" data-act="edit-person">Edit details</button>
      <button class="small danger" data-act="delete-person">Delete</button>
      <span class="status error" id="person-error" role="alert"></span>
    </div>
    ${state.editing === 'person' ? personForm(p) : ''}

    <div class="block">
      <h4>Also called</h4>
      ${p.aliases.map((a) => `<span class="chip">${esc(a)}<button data-act="drop-alias"
        data-alias="${esc(a)}">&times;</button></span>`).join('') || '<span class="empty">nothing yet</span>'}
      <div class="inline">
        <input type="text" id="new-alias" placeholder="another name for them">
        <button class="small" data-act="add-alias">Add</button>
      </div>
    </div>

    <div class="block">
      <h4>Groups</h4>
      ${p.groups.map((g) => `<span class="chip">${esc(g.name)}<button data-act="leave-group"
        data-id="${g.id}">&times;</button></span>`).join('') || '<span class="empty">none</span>'}
      ${joinable.length ? `<div class="inline">
        <select id="join-group">${joinable.map((g) =>
          `<option value="${g.id}">${esc(g.name)}</option>`).join('')}</select>
        <button class="small" data-act="join-group">Add to group</button>
      </div>` : ''}
    </div>

    <div class="block">
      <h4>Facts</h4>
      ${p.facts.map((f) => factRow(f, false)).join('') || '<p class="empty">nothing yet</p>'}
      <div class="inline">
        <input type="text" id="new-fact" placeholder="something true about them">
        <button class="small" data-act="add-fact">Add</button>
      </div>
    </div>

    ${p.commitments.length ? `<div class="block">
      <h4>Open</h4>
      ${p.commitments.map((c) => `<div class="fact">
        <div><div class="kind">${c.direction === 'i_owe' ? 'You owe them' : 'They owe you'}</div>
        <div>${esc(c.what)}${c.due_date ? ` (due ${esc(c.due_date)})` : ''}</div></div>
        <div class="actions">
          <button class="small" data-act="settle" data-id="${c.id}" data-status="done">Done</button>
          <button class="small" data-act="settle" data-id="${c.id}" data-status="dropped">Drop</button>
        </div></div>`).join('')}
    </div>` : ''}

    ${p.dates.length ? `<div class="block">
      <h4>Dates</h4>
      ${p.dates.map((d) => `<div class="fact">
        <div>${esc(d.label)}: ${esc(d.date)}${d.recurring ? ', every year' : ''}</div>
        <div class="actions"><button class="small" data-act="drop-date"
          data-id="${d.id}">Remove</button></div></div>`).join('')}
    </div>` : ''}

    ${p.history.length ? `<div class="block">
      <h4>History</h4>
      ${state.history ? p.history.map((f) => factRow(f, true)).join('')
        : `<button class="small" data-act="show-history">Show ${p.history.length} older</button>`}
    </div>` : ''}

    ${others.length ? `<div class="block">
      <h4>Same person as</h4>
      <div class="inline">
        <select id="merge-with">${others.map((o) =>
          `<option value="${o.id}">${esc(o.name)}</option>`).join('')}</select>
        <button class="small danger" data-act="merge">Merge into this one</button>
      </div>
    </div>` : ''}`;
}

function personForm(p) {
  const field = (key, label, value, type = 'text') =>
    `<label for="p-${key}">${label}</label>` + (type === 'area'
      ? `<textarea id="p-${key}" rows="2">${esc(value || '')}</textarea>`
      : `<input type="text" id="p-${key}" value="${esc(value || '')}">`);
  return `<div class="edit" style="margin-top:12px">
    ${field('name', 'Name', p.name)}
    ${field('relationship', 'Relationship', p.relationship)}
    <label for="p-closeness">Closeness</label>
    <select id="p-closeness">${Object.entries(CLOSENESS).map(([v, label]) =>
      `<option value="${v}"${Number(v) === p.closeness ? ' selected' : ''}>${esc(label)}</option>`).join('')}</select>
    ${field('how_met', 'How you met', p.how_met)}
    ${field('summary', 'Who they are', p.summary, 'area')}
    <label for="p-status">Status</label>
    <select id="p-status">${['active', 'dormant', 'lost_touch'].map((v) =>
      `<option value="${v}"${v === p.status ? ' selected' : ''}>${esc(v.replace('_', ' '))}</option>`).join('')}</select>
    <span></span>
    <span><button class="small primary" data-act="save-person">Save</button>
      <button class="small" data-act="cancel-person">Cancel</button></span>
  </div>`;
}

// -- acting on a person -----------------------------------------------------

async function act(fn) {
  try {
    await fn();
    await loadDirectory(true);
  } catch (error) {
    const status = $('person-error');
    if (status) status.textContent = error.message;
    else $('dir-status').textContent = error.message;
  }
}

document.addEventListener('click', async (event) => {
  const el = event.target.closest('[data-act]');
  if (!el || !$('people-view') || $('people-view').hidden) return;
  const act_ = el.dataset.act;
  const id = el.dataset.id;
  const p = state.person;

  if (act_ === 'open') return openPerson(id);
  if (act_ === 'source') {
    state.showing = state.showing === id ? null : id;
    return renderProfile();
  }
  if (act_ === 'show-history') { state.history = true; return renderProfile(); }
  if (act_ === 'edit-person') { state.editing = 'person'; return renderProfile(); }
  if (act_ === 'edit-fact') { state.editing = id; return renderProfile(); }
  if (act_ === 'cancel-person' || act_ === 'cancel-fact') {
    state.editing = null;
    return renderProfile();
  }

  if (act_ === 'drop-group') {
    if (!confirm('Remove this group? The people in it stay.')) return;
    return act(() => api(`/people/groups/${id}`, undefined, 'DELETE'));
  }
  if (!p) return;

  if (act_ === 'save-person') {
    return act(() => api(`/people/person/${p.id}`, {
      name: $('p-name').value.trim(),
      relationship: $('p-relationship').value.trim() || null,
      closeness: Number($('p-closeness').value),
      how_met: $('p-how_met').value.trim() || null,
      summary: $('p-summary').value.trim() || null,
      status: $('p-status').value,
    }, 'PATCH'));
  }
  if (act_ === 'delete-person') {
    if (!confirm(`Delete ${p.name} and everything about them?`)) return;
    state.person = null;
    return act(() => api(`/people/person/${p.id}`, undefined, 'DELETE'));
  }
  if (act_ === 'add-alias') {
    const alias = $('new-alias').value.trim();
    if (alias) return act(() => api(`/people/person/${p.id}/alias`, { alias }));
    return;
  }
  if (act_ === 'drop-alias') {
    return act(() => api(`/people/person/${p.id}/alias`,
      { alias: el.dataset.alias, present: false }));
  }
  if (act_ === 'join-group') {
    return act(() => api(`/people/groups/${$('join-group').value}/members`,
      { person_id: p.id }));
  }
  if (act_ === 'leave-group') {
    return act(() => api(`/people/groups/${id}/members`,
      { person_id: p.id, member: false }));
  }
  if (act_ === 'add-fact') {
    const text = $('new-fact').value.trim();
    if (text.length > 2) return act(() => api(`/people/person/${p.id}/facts`, { text }));
    return;
  }
  if (act_ === 'save-fact') {
    return act(() => api(`/people/facts/${id}`, {
      category: $('e-cat').value,
      text: $('e-text').value.trim(),
      sensitivity: $('e-sens').value,
    }, 'PATCH'));
  }
  if (act_ === 'retire-fact') {
    if (!confirm('Retire this fact? It stays readable in history.')) return;
    return act(() => api(`/people/facts/${id}`, undefined, 'DELETE'));
  }
  if (act_ === 'settle') {
    return act(() => api(`/people/commitments/${id}`, { status: el.dataset.status }));
  }
  if (act_ === 'drop-date') {
    return act(() => api(`/people/dates/${id}`, undefined, 'DELETE'));
  }
  if (act_ === 'merge') {
    const other = $('merge-with');
    const name = other.options[other.selectedIndex].text;
    if (!confirm(`Merge ${name} into ${p.name}? Everything moves here.`)) return;
    return act(() => api('/people/merge', { keep: p.id, merge: other.value }));
  }
});
