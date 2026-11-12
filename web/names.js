// Plain-language names for weakness-map tags (shared by the play and weakness pages).
export const NAMES = {
  "position: winning": "Converting winning positions",
  "position: balanced": "Balanced positions",
  "position: losing": "Defending worse positions",
  "material: ahead": "When you're up material",
  "material: even": "Material even",
  "material: behind": "When you're down material",
  "phase: opening": "Opening (moves 1–10)",
  "phase: middlegame": "Middlegame",
  "phase: endgame": "Endgames",
  "clock: time trouble (<10% or <30s)": "Time trouble",
  "clock: enough time": "With time on the clock",
  "situation: in check": "Getting out of check",
  "situation: a piece under attack": "When one of your pieces is attacked",
  "situation: king still in the center": "King still in the center",
  "mistake type: hung material": "Hanging material",
  "mistake type: allowed mate": "Walking into mate",
  "mistake type: missed mate": "Missing mates",
  "mistake type: allowed fork": "Walking into forks",
  "mistake type: missed material": "Missing free material",
  "mistake type: missed fork": "Missing forks",
};
export const nameOf = (tag) => {
  if (NAMES[tag]) return NAMES[tag];
  const [dim, val] = tag.split(": ");
  if (dim === "piece moved") return `${val[0].toUpperCase()}${val.slice(1)} moves`;
  if (dim === "endgame type") return `${val[0].toUpperCase()}${val.slice(1)}s`;
  return tag;
};

