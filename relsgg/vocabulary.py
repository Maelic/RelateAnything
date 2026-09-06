"""Default predicate vocabulary.

The predicate set is an input to the model, not a property of it; this list
is the one the released bundles ship and the demo uses. Synonyms are kept
apart on purpose (``on`` / ``resting on`` / ``on top of``): the model was
trained on a synonym-rich vocabulary and they compete at decode time.
"""

DEFAULT_PREDICATES = [
    # people doing things
    "wearing", "riding", "playing", "sitting on", "sitting at", "holding",
    "sitting in", "looking at", "using", "watching", "standing on",
    "carrying", "talking to", "smiling at", "standing beside", "walking past",
    "posing with", "leaning against",
    # object-object structure
    "part of", "resting on", "on", "covering", "inside", "on top of",
    "contained in", "hanging from", "surrounding", "attached to",
    # spatial frame
    "in front of", "beside", "to the left of", "to the right of", "behind",
    "above", "below",
]

# The template ensemble the vocabulary was encoded with during training. A
# predicate's embedding is the normalised mean over these three phrasings.
TRAIN_TEMPLATES = ["{p}", "one object is {p} another object",
                   "a photo of something {p} something"]
