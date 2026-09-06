import os
import shutil
import torch


# ============================================================
# CHECKPOINT
# ============================================================

checkpoint_path = os.path.join( "Version4_continuous_windows", "checkpoint.pt" )

backup_path = os.path.join( "Version4_continuous_windows", "checkpoint_before_window_change.pt")

# ============================================================
# LOAD CHECKPOINT
# ============================================================

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError( f"Checkpoint not found: {checkpoint_path}" )


# Make backup before changing anything
shutil.copy2( checkpoint_path, backup_path)

print()
print("Backup created:")
print(backup_path)
print()


checkpoint = torch.load( checkpoint_path, map_location="cpu", weights_only=False)


local_states = checkpoint["local_model_state_dicts"]

print( "Number of local windows:", len(local_states))

print()


# ============================================================
# READ AND MODIFY EACH WINDOW
# ============================================================

for i, state in enumerate(local_states):

    # --------------------------------------------------------
    # Current values
    # --------------------------------------------------------

    xL_old = state["xL"].item()

    width_fraction_old = ( state["width_fraction"].item() )

    xR_old = ( xL_old + (1.0 - xL_old) * width_fraction_old )


    print("=" * 60)
    print(f"WINDOW {i + 1}")
    print("=" * 60)

    print( f"Current xL = {xL_old:.8f}")

    print( f"Current xR = {xR_old:.8f}" )

    print()


    # --------------------------------------------------------
    # Get new values
    # --------------------------------------------------------

    xL_input = input( f"Enter new xL for Window {i + 1} "
                      f"(press Enter to keep {xL_old:.8f}): ").strip()

    xR_input = input( f"Enter new xR for Window {i + 1} "
                      f"(press Enter to keep {xR_old:.8f}): ").strip()


    if xL_input == "":
        xL_new = xL_old
    else:
        xL_new = float(xL_input)


    if xR_input == "":
        xR_new = xR_old
    else:
        xR_new = float(xR_input)


    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not ( 0.0 < xL_new < xR_new < 1.0):
        raise ValueError(
            f"Invalid Window {i + 1}: "
            f"must satisfy "
            f"0 < xL < xR < 1."
        )


    # --------------------------------------------------------
    # Convert xR into width_fraction
    #
    # xR = xL + (1-xL)*width_fraction
    #
    # therefore
    #
    # width_fraction =
    # (xR-xL)/(1-xL)
    # --------------------------------------------------------

    width_fraction_new = ( (xR_new - xL_new)/ (1.0 - xL_new))


    # --------------------------------------------------------
    # Write new values into state_dict
    # --------------------------------------------------------

    state["xL"] = torch.tensor( [[xL_new]], dtype=state["xL"].dtype)

    state["width_fraction"] = torch.tensor( [[width_fraction_new]], dtype=state["width_fraction"].dtype)


    print()

    print( f"New xL = {xL_new:.8f}")

    print( f"New xR = {xR_new:.8f}")

    print(f"New width_fraction = "
          f"{width_fraction_new:.8f}")

    print()


# ============================================================
# SAVE MODIFIED CHECKPOINT
# ============================================================

checkpoint["local_model_state_dicts"] = ( local_states)

torch.save( checkpoint, checkpoint_path)


print("=" * 60)
print("CHECKPOINT UPDATED")
print("=" * 60)

print( "Modified checkpoint:", checkpoint_path)

print( "Original checkpoint backup:", backup_path)

print()


# ============================================================
# VERIFY SAVED VALUES
# ============================================================

checkpoint_check = torch.load( checkpoint_path, map_location="cpu", weights_only=False)

local_states_check = ( checkpoint_check["local_model_state_dicts"])


print("VERIFYING SAVED WINDOW VALUES")
print()

for i, state in enumerate( local_states_check):

    xL = state["xL"].item()

    width_fraction = ( state["width_fraction"].item() )

    xR = ( xL + (1.0 - xL) * width_fraction )

    print(
        f"Window {i + 1}: "
        f"xL = {xL:.8f}, "
        f"xR = {xR:.8f}"
    )