
ATTITUDE_INSTRUCTIONS = [
    "You are given a dialogue between the therapist and the client.\n{dialogue}\nPredict the client attitude. "
    "Options are \"change\" (motivation toward behaviour change), "
    "\"neutral\" (neutral attitude or not enough information) "
    "or \"sustain\" (resistance against behaviour change).",  # default choice
    "Consider this dialogue between the therapist and the client:\n{dialogue}\nWhat is the client attitude? "
    "Choose \"change\" if the client is motivated toward behaviour change, "
    "\"sustain\" if the client shows resistance against behaviour change, or "
    "\"neutral\" if no information is available.",
    "{dialogue}\nHow would one describe the client attitude in the dialogue: "
    "\"change\" (motivation toward behaviour change), \"sustain\" (resistance against behaviour change), "
    "or \"neutral\" (neutral attitude or not enough information)."
]

ATTITUDE_SIMPLE_INSTRUCTIONS = [
    "You are given a dialogue between the therapist and the client.\n{dialogue}\nPredict the client attitude. "
    "Options are \"change\", \"neutral\", or \"sustain\".",
]

ATTITUDE_INSTRUCTIONS_WITH_ICL = [
    "You are given a dialogue between the therapist and the client. Predict the client attitude. "
    "Options are \"change\" (motivation toward behaviour change), \"sustain\" (resistance against behaviour change), "
    "or \"neutral\" (neutral attitude or not enough information). "
    "{samples}\n\nNow, predict the client attitude in the following dialogue:\n{dialogue}",
]

ATTITUDE_SIMPLE_INSTRUCTIONS_WITH_ICL = [
    "You are given a dialogue between the therapist and the client. Predict the client attitude. "
    "Options are \"change\", \"sustain\", or \"neutral\". "
    "{samples}\n\nNow, predict the client attitude in the following dialogue:\n{dialogue}",
]

STRENGTH_INSTRUCTIONS = [
    "What is the certainty level of the client attitude in the dialogue between the therapist and the client? "
    "Choose \"high\", \"medium\", or \"low\".\n{dialogue}"
]

STRENGTH_INSTRUCTIONS_WITH_ICL = [
    "What is the certainty level of the client attitude in the dialogue between the therapist and the client? "
    "Choose \"high\", \"medium\", or \"low\". "
    "{samples}\n\nNow, predict the certainty level in the following dialogue:\n{dialogue}",
]

ATTITUDE_STRENGTH_INSTRUCTIONS = [
    "You are given a dialogue between the therapist and the client.\n{dialogue}"
    "\nPredict the client attitude and their certainty level. "
    "Choose \"(a) change high\" (high motivation toward behaviour change), "
    "\"(b) change medium\" (medium motivation toward behaviour change), "
    "\"(c) change low\" (low motivation toward behaviour change), "
    "\"(d) neutral high\" (high neutral attitude), "
    "\"(e) neutral medium\" (medium neutral attitude), "
    "\"(f) neutral low\" (low neutral attitude), "
    "\"(g) sustain high\" (high resistance against behaviour change), "
    "\"(h) sustain medium\" (medium resistance against behaviour change), "
    "or \"(i) sustain low\" (low resistance against behaviour change). "
]

ATTITUDE_STRENGTH_SIMPLE_INSTRUCTIONS = [
    "You are given a dialogue between the therapist and the client.\n{dialogue}"
    "\nPredict the client attitude and their certainty level. "
    "Choose \"(a) change high\", "
    "\"(b) change medium\", "
    "\"(c) change low\", "
    "\"(d) neutral high\", "
    "\"(e) neutral medium\", "
    "\"(f) neutral low\", "
    "\"(g) sustain high\", "
    "\"(h) sustain medium\", "
    "or \"(i) sustain low\". "
]

ATTITUDE_STRENGTH_INSTRUCTIONS_WITH_ICL = [
    "You are given a dialogue between the therapist and the client. "
    "Predict the client attitude and their certainty level. "
    "Choose \"(a) change high\" (high motivation toward behaviour change), "
    "\"(b) change medium\" (medium motivation toward behaviour change), "
    "\"(c) change low\" (low motivation toward behaviour change), "
    "\"(d) neutral high\" (high neutral attitude), "
    "\"(e) neutral medium\" (medium neutral attitude), "
    "\"(f) neutral low\" (low neutral attitude), "
    "\"(g) sustain high\" (high resistance against behaviour change), "
    "\"(h) sustain medium\" (medium resistance against behaviour change), "
    "or \"(i) sustain low\" (low resistance against behaviour change). "
    "{samples}\n\nNow, predict the client attitude and their certainty level in the following dialogue:\n{dialogue}"
]

ATTITUDE_STRENGTH_SIMPLE_INSTRUCTIONS_WITH_ICL = [
    "You are given a dialogue between the therapist and the client. "
    "Predict the client attitude and their certainty level. "
    "Choose \"(a) change high\", "
    "\"(b) change medium\", "
    "\"(c) change low\", "
    "\"(d) neutral high\", "
    "\"(e) neutral medium\", "
    "\"(f) neutral low\", "
    "\"(g) sustain high\", "
    "\"(h) sustain medium\", "
    "or \"(i) sustain low\". "
    "{samples}\n\nNow, predict the client attitude and their certainty level in the following dialogue:\n{dialogue}"
]


def get_prompt_for_task(task: str, therapist_utt: str, client_utt: str, target_space: str,
                        use_therapist_utt, in_context_text="", use_simple_inst=False) -> str:
    target_space = target_space.strip().split(' ')
    dialogue = f"Therapist: {therapist_utt}\nClient: \"{client_utt}\"" if len(therapist_utt) > 0 and use_therapist_utt \
        else f"Client: \"{client_utt}\""

    if len(in_context_text) > 0:
        if task == 'attitude':
            return get_attitude_icl_instruction(dialogue, in_context_text, use_simple_inst)
        elif task == 'certainty':
            return get_certainty_icl_instruction(dialogue, in_context_text) if len(target_space) == 1 \
                else get_attitude_certainty_icl_instruction(dialogue, in_context_text, use_simple_inst)
        elif task == 'multitask':
            return get_certainty_icl_instruction(dialogue, in_context_text) if len(target_space) == 1 \
                else get_attitude_certainty_icl_instruction(dialogue, in_context_text, use_simple_inst)
    else:
        if task == 'attitude':
            return get_attitude_instruction(dialogue, use_simple_inst)
        elif task == 'certainty':
            return get_certainty_instruction(dialogue) if len(target_space) == 1 \
                else get_attitude_certainty_instruction(dialogue, use_simple_inst)
        elif task == 'multitask':
            return get_certainty_instruction(dialogue) if len(target_space) == 1 \
                else get_attitude_certainty_instruction(dialogue, use_simple_inst)


def get_attitude_instruction(dialogue: str, use_simple_inst: bool) -> str:
    # return random.choice(ATTITUDE_INSTRUCTIONS).format(**{'dialogue': dialogue})
    return ATTITUDE_SIMPLE_INSTRUCTIONS[0].format(**{'dialogue': dialogue}) if use_simple_inst \
        else ATTITUDE_INSTRUCTIONS[0].format(**{'dialogue': dialogue})


def get_attitude_icl_instruction(dialogue: str, samples: str, use_simple_inst: bool) -> str:
    return ATTITUDE_SIMPLE_INSTRUCTIONS_WITH_ICL[0].format(**{'samples': samples, 'dialogue': dialogue}) \
        if use_simple_inst else ATTITUDE_INSTRUCTIONS_WITH_ICL[0].format(**{'samples': samples, 'dialogue': dialogue})


def get_certainty_instruction(dialogue: str) -> str:
    return STRENGTH_INSTRUCTIONS[0].format(**{'dialogue': dialogue})


def get_certainty_icl_instruction(dialogue: str, samples: str) -> str:
    return STRENGTH_INSTRUCTIONS_WITH_ICL[0].format(**{'samples': samples, 'dialogue': dialogue})


def get_attitude_certainty_instruction(dialogue: str, use_simple_inst: bool) -> str:
    return ATTITUDE_STRENGTH_SIMPLE_INSTRUCTIONS[0].format(**{'dialogue': dialogue}) \
        if use_simple_inst else ATTITUDE_STRENGTH_INSTRUCTIONS[0].format(**{'dialogue': dialogue})


def get_attitude_certainty_icl_instruction(dialogue: str, samples: str, use_simple_inst: bool) -> str:
    return ATTITUDE_STRENGTH_SIMPLE_INSTRUCTIONS_WITH_ICL[0].format(**{'samples': samples, 'dialogue': dialogue}) \
        if use_simple_inst \
        else ATTITUDE_STRENGTH_INSTRUCTIONS_WITH_ICL[0].format(**{'samples': samples, 'dialogue': dialogue})
