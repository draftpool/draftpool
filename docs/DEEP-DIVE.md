# Deep dive

The fine print behind the [guide](GUIDE.md): limits you'll only run into in
particular situations, and what to do when you do. None of it is needed to use
the app.

## Edit: what it can and cannot change

The basics are in the [guide](GUIDE.md#edit-changing-a-picture). This section
covers what to do when an edit doesn't do what you asked.

**"Another take on it" keeps the composition and redraws the detail.** The
layout, framing, pose and props stay; folds, hands, hems and the exact crop
change. It runs at a fixed strength of 0.8. Lower strengths were tried and
changed nothing you asked for (a dress colour, the time of day, the style), so
there is no slider.

**Change what's there can't add something new.** Repainting keeps what is
already in the rectangle, so asking for a dog on empty grass gives you grass. Use **add
something new** instead, which sends the rectangle to the instruction editor.

**Everything outside your rectangle is left exactly as it was**, pixel for
pixel. Inside it, you get whatever the editor drew, so a slight change in tone
can appear inside the rectangle too.

**The new object is drawn at the size of your rectangle.** The editor only sees
what is inside the rectangle, so if the rectangle is too small, the object will
be cropped. Draw it around all the space the object needs.

**Tall or wide rectangles work.** The render matches your rectangle's shape up
to about 4:1. Beyond that the editor gets a slightly wider frame, and part of
what it draws may be cut off.

**Big rectangles come back softer.** The model repaints a window about 2.5
times the size of your rectangle at 1024px, so a rectangle over about 400px on
its long side is repainted at lower detail than the rest of the picture. The
dialog warns you. For a large area, edit the whole image instead, or use
several smaller rectangles.

**Word the change the way the model reads it.** An anime model reads *replace
the cats with birds* as the tags "cats" and "birds" and may produce neither.
With a rectangle, write what should be there: `black cat`. Instructions like
"replace…" belong to the no-rectangle route, which uses a model trained to
follow them.

## Editing an upscaled image

An edit without a rectangle renders at the edit model's working size, not at
the size of your image. Editing a 4096px upscale with Qwen Edit gives you back
roughly 1024px (or the same number of pixels in your image's shape, e.g.
1344×768 for 16:9). The dialog shows the output size before you start.

Ways around it:

- **Edit first, then upscale.** The simplest option and usually the best.
- **Upscale** or **2x fast** the edited result to get the size back.
- **Use a rectangle.** Only the rectangle is repainted, so the rest of the image
  keeps its full resolution.
- **Use another take on it and tick "repaint at N×N instead"** to edit at the
  image's full size. Slower, and the dialog shows how much. An edit by
  instruction doesn't offer it: Qwen Edit at full size would take tens of
  minutes.

Not affected: **Fix faces** works on the faces alone, so the image keeps its
size. It skips a face that is already large (common on an upscale), because
repainting it would add nothing, and says so. **More
like this** makes new images. **Animate** always renders at about 832×480.

## Upscaling with a different model

By default an image is upscaled by the model that made it. You can choose
another in the picker next to the button, for a change of style or a faster
model, but expect the fine detail to change: softer faces, blurrier foliage.

## Why a pose change never animates

The video model moves the pixels of the first frame. It has no idea of a body
or skeleton, so no wording will make a standing person sit down. You'll get a
smooth clip where they shift or take a step instead. Start from a still that
already has the pose you want.

It also doesn't understand negatives, which is why "the camera stays still"
makes the camera move: it acts on the words that are there.

## The tag vocabulary in detail

The [guide](GUIDE.md#prompting-the-anime-models) explains what the tag checks
do. Here is what the suggestions are made of.

**If a word isn't a tag but there is a tag that means it**, the composer offers
it. Type **male** and you're offered **1boy**, **male focus** and **multiple
boys**; **female** offers **1girl**. You choose, because only you know which
you meant.

**A plain adjective gets the tags that use it.** Danbooru tags *what* is long
or open rather than the quality itself, so **long** offers **long hair** and
**long sleeves**, and likewise **open**, **bare**, **wide** and **holding**.

**Some words have no tag at all.** These come back with nothing offered. Words
like **beautiful**, **gorgeous** and **stunning** are already covered: every
prompt to these models starts with quality words (`masterpiece, best
quality`), and adding more only weakens the words that describe the picture.

**Quality words aren't Danbooru tags** but the models were trained on them, so
they're left alone.

**Some tags have a preferred spelling.** Danbooru treats `mist` as another name
for `fog`, and `aerial view` for `from above`. The composer suggests the main
spelling, and only swaps it for you when the whole phrase is the tag.

**Unknown words are never deleted.** "green raincoat" isn't a tag, but cutting
it to "raincoat" would lose the colour, so it's left as written.

The tag list is in `config/danbooru_tags.json`. To refresh it from Danbooru,
run `uv run python scripts/fetch_tags.py`.

## Prompt labels

Images from your own prompt are labelled **orig**; the LLM's prompts are
**V1**, **V2** and so on (or **T1**/**P1** when tag-based and sentence-based
models are selected together).

The numbers keep counting across the whole session, so a second Enrich
continues at V4. That way a label always refers to one prompt, even after the
composer shows newer ones.

## When Draw Things goes away, or stops rendering

**Jobs wait rather than fail.** If Draw Things is closed or its API Server
setting has switched off, queued jobs wait until it answers again.

Sometimes Draw Things keeps reporting success but returns no image, or starts
drawing everything twice (two heads, a second horizon). The app notices: the
header says "Draw Things not rendering" and the queue waits. **Restart Draw
Things** and the queue carries on by itself.

If a job fails again straight after Draw Things recovers, the problem is with
that job (for example, a size the model doesn't support), and the message says
so.

If images suddenly look wrong on models that were fine earlier, restart Draw
Things before blaming the model or your prompt.

## When the LLM fails

You get a message with a **Retry** button and the reason: Ollama isn't running,
the model gave an answer the app couldn't use, or it didn't answer in time.

The time limit is five minutes (`llm.request_timeout` in `config/models.yaml`).
A large model can take longer on its first request while it loads, so **Retry**
often works. If it keeps timing out, choose a smaller model. See also
[Setup](SETUP.md#troubleshooting).
