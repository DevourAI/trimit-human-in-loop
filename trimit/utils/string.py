import string


def strip_punctuation_case_whitespace(w):
    return w.strip().lower().strip(" \"'" + string.punctuation)


def longest_contiguous_match(list1, list2):
    # Initialize variables to store the longest match found
    max_length = 0
    start_index = 0

    # Iterate through list1
    for i in range(len(list1)):
        # Iterate through list2
        for j in range(len(list2)):
            # Track the current match length
            match_length = 0
            # While we're within the bounds of both lists and the items match...
            while (
                i + match_length < len(list1)
                and j + match_length < len(list2)
                and list1[i + match_length] == list2[j + match_length]
            ):
                # Increment the match length
                match_length += 1
                # If this is the longest match we've found so far...
                if match_length > max_length:
                    # Update the longest match details
                    max_length = match_length
                    start_index = j

    # If there was a match, calculate the end index
    end_index = start_index + max_length if max_length > 0 else start_index
    # Return the start and end index of the longest match
    return (start_index, end_index)
