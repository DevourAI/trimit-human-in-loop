from modal.functions import FunctionCall, gather
from modal.call_graph import InputStatus


async def wait_for_completion(*call_ids):
    fcs = [FunctionCall.from_id(call_id) for call_id in call_ids]
    gather(*fcs)

    cgs = []
    for fc in fcs:
        cgs.extend(fc.get_call_graph())
    failures = []
    successes = []
    stack = cgs[:]
    seen = set()
    while len(stack):
        node = stack.pop()
        if node.task_id in seen:
            continue
        seen.add(node.task_id)
        if node.status == InputStatus.FAILURE:
            failures.append(node)
        else:
            successes.append(node)
        for child in node.children:
            stack.append(child)
    print(f"Found {len(failures)} failures")
    for f in failures:
        print(f)
    print(f"Found {len(successes)} successes")
    for s in successes:
        print(s)
