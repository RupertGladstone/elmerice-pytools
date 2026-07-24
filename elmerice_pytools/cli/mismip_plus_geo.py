#!/usr/bin/env python3
"""Generate a Gmsh .geo file for the MISMIP+ domain (Asay-Davis et al., 2016;
Cornford et al., 2020).

Coordinates are absolute MISMIP+ coordinates: x=0 is the ice divide and
x=640 km is the nominal calving front in the default-size domain; the bedrock
formula (Eq. 1 of Cornford et al., 2020) is defined in this same absolute
frame. To make room for extra ice volume upstream without changing the
bedrock/calving-front geometry over the original active region, --lx grows
the domain by extending it in the *negative* x direction: the calving front
stays pinned at x=640 km and the inflow (no-slip) boundary moves upstream to
x = 640 km - lx. The default --lx=640e3 reproduces x in [0, 640] km exactly.
--ly grows the domain symmetrically about y=0, since the channel bedrock is
mirror-symmetric across the centerline.

Mesh resolution is uniform (dx-background) away from the grounding-line zone,
and refined (dx-refined) in a band [refine-xmin, refine-xmax] (absolute x,
unaffected by domain size) spanning the full channel width, with a smooth
transition over a distance of --transition.

Optionally, --centerline-scale further scales the mesh size in a strip
centered on y=0 (e.g. for resolving subglacial outflow/plumes at the
grounding line), spanning the full x-extent of the domain. Within
|y| <= centerline-yin the local size (dx_refined inside the x-refined band,
dx_background outside it) is multiplied by centerline_scale; the scale factor
relaxes linearly back to 1 (no effect) by |y| = centerline-yout. Default
centerline_scale=1 disables this (no effect, no extra mesh field).

Boundary physical IDs (referenced from the .sif file as Target Boundaries):
    1  inflow         x = x_min         (no-slip)
    2  calving_front  x = 640 km        (calving front / open boundary)
    3  lateral_south  y = y_min         (free-slip wall)
    4  lateral_north  y = y_max         (free-slip wall)
Body physical ID:
    1  ice
"""
import argparse
import os

ACTIVE_X_MAX = 640e3  # calving front position in the MISMIP+ coordinate frame

# MISMIP+ bedrock parameters, Eq. (1) and Table 1 of Cornford et al. (2020)
BEDROCK_PARAMS = dict(
    x_bar=300e3, B0=-150.0, B2=-728.8, B4=343.91, B6=-50.75,
    w_c=24e3, f_c=4e3, d_c=500.0,
)


def bedrock_elevation(x, y, x_bar, B0, B2, B4, B6, w_c, f_c, d_c):
    """MISMIP+ bedrock elevation z_b(x, y), Eq. (1) of Cornford et al. (2024)."""
    import numpy as np
    b_x = B0 + B2 * (x / x_bar) ** 2 + B4 * (x / x_bar) ** 4 + B6 * (x / x_bar) ** 6
    b_y = d_c * (
        1.0 / (1.0 + np.exp(-2.0 * (y - w_c) / f_c))
        + 1.0 / (1.0 + np.exp(2.0 * (y + w_c) / f_c))
    )
    return np.maximum(b_x + b_y, -720.0)


def plot_bedrock(x_min, x_max, y_min, y_max, refine_xmin, refine_xmax, output_path,
                  n_x=400, n_y=200, max_xy_aspect=3.0):
    """Render a 3D surface of the MISMIP+ bedrock over the given domain and
    save it to output_path. Also marks the mesh-refinement band edges.

    The x and y axes are scaled to their true relative proportions (so a
    wide, narrow domain looks wide and narrow rather than square), but the
    x:y ratio is capped at max_xy_aspect so extreme domains (e.g. the default
    640x80 km) don't render as an unreadably thin sliver.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.linspace(x_min, x_max, n_x)
    y = np.linspace(y_min, y_max, n_y)
    X, Y = np.meshgrid(x, y)
    Z = bedrock_elevation(X, Y, **BEDROCK_PARAMS)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X / 1e3, Y / 1e3, Z, cmap="terrain",
                            linewidth=0, antialiased=True, rstride=2, cstride=2)

    xy_aspect = min((x_max - x_min) / (y_max - y_min), max_xy_aspect)
    ax.set_box_aspect((xy_aspect, 1, 1))

    # lift marker lines above the surface so they aren't z-fighting with it; a
    # fixed 1m offset is imperceptible next to O(100-1000m) of bedrock relief,
    # so scale it to the surface's own elevation range instead
    marker_z_offset = 0.02 * (Z.max() - Z.min())

    y_line = np.linspace(y_min, y_max, n_y)
    for x_edge, label in ((refine_xmin, "refine_xmin"), (refine_xmax, "refine_xmax")):
        if x_min <= x_edge <= x_max:
            z_line = bedrock_elevation(np.full_like(y_line, x_edge), y_line, **BEDROCK_PARAMS)
            ax.plot(np.full_like(y_line, x_edge / 1e3), y_line / 1e3, z_line + marker_z_offset,
                    color="red", linestyle="--", linewidth=2, label=label)

    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("bedrock elevation (m)")
    ax.set_title(f"MISMIP+ bedrock, x in [{x_min/1e3:.0f}, {x_max/1e3:.0f}] km, "
                 f"y in [{y_min/1e3:.0f}, {y_max/1e3:.0f}] km")
    fig.colorbar(surf, shrink=0.6, label="z_b (m)")
    ax.legend(loc="upper left")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


GEO_TEMPLATE = """\
// MISMIP+ domain -- generated by elmerice_pytools.cli.mismip_plus_geo
// x in [{x_min}, {x_max}], y in [{y_min}, {y_max}]
// dx_background = {dx_background}, dx_refined = {dx_refined}
// refined band: x in [{refine_xmin}, {refine_xmax}], transition = {transition}

Point(1) = {{{x_min}, {y_min}, 0, {dx_background}}};
Point(2) = {{{x_max}, {y_min}, 0, {dx_background}}};
Point(3) = {{{x_max}, {y_max}, 0, {dx_background}}};
Point(4) = {{{x_min}, {y_max}, 0, {dx_background}}};

Line(1) = {{1, 2}};  // y = y_min, lateral wall (free-slip)
Line(2) = {{2, 3}};  // x = x_max, calving front
Line(3) = {{3, 4}};  // y = y_max, lateral wall (free-slip)
Line(4) = {{4, 1}};  // x = x_min, inflow (no-slip)

Line Loop(1) = {{1, 2, 3, 4}};
Plane Surface(1) = {{1}};

Physical Line(1) = {{4}};    // inflow
Physical Line(2) = {{2}};    // calving_front
Physical Line(3) = {{1}};    // lateral_south
Physical Line(4) = {{3}};    // lateral_north
Physical Surface(1) = {{1}}; // ice

// Mesh sizing: ignore point/curve-based sizing, use the Box field only
Mesh.CharacteristicLengthFromPoints = 0;
Mesh.CharacteristicLengthFromCurvature = 0;
Mesh.CharacteristicLengthExtendFromBoundary = 0;
Mesh.CharacteristicLengthMin = {length_min};
Mesh.CharacteristicLengthMax = {dx_background};
Mesh.Algorithm = 6; // Frontal-Delaunay

Field[1] = Box;
Field[1].VIn = {dx_refined};
Field[1].VOut = {dx_background};
Field[1].XMin = {refine_xmin};
Field[1].XMax = {refine_xmax};
Field[1].YMin = {y_min};
Field[1].YMax = {y_max};
Field[1].Thickness = {transition};
{centerline_field}"""

CENTERLINE_FIELD_TEMPLATE = """
// Field[2] is a dimensionless scale factor (not a size): {centerline_scale}
// within |y|<=centerline-yin, relaxing linearly to 1 (no effect) by
// |y|=centerline-yout. Field[3] multiplies it onto Field[1]'s size, so the
// centerline scaling shrinks whatever size would otherwise apply there
// (dx_refined inside the x-refined band, dx_background outside it) rather
// than substituting a fixed size of its own.
Field[2] = Box;
Field[2].VIn = {centerline_scale};
Field[2].VOut = 1.0;
Field[2].XMin = {x_min};
Field[2].XMax = {x_max};
Field[2].YMin = {centerline_yin_neg};
Field[2].YMax = {centerline_yin};
Field[2].Thickness = {centerline_thickness};

Field[3] = MathEval;
Field[3].F = "F1*F2";
Background Field = 3;
"""

NO_CENTERLINE_FIELD = """
Background Field = 1;
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lx", type=float, default=640e3,
                    help="total domain length in x (default: 640 km, enter as "
                         "metres). Extra length beyond 640 km is added upstream "
                         "(negative x); the calving front stays fixed at x=640 km")
    p.add_argument("--ly", type=float, default=80e3,
                    help="total domain length in y (default: 80 km, enter as "
                         "metres), added symmetrically about y=0")
    p.add_argument("--dx-background", type=float, default=3000.0,
                    help="mesh size away from the refined band "
                         "(default: 3 km, enter as metres)")
    p.add_argument("--dx-refined", type=float, default=1000.0,
                    help="mesh size inside the refined band "
                         "(default: 1 km, enter as metres)")
    p.add_argument("--refine-xmin", type=float, default=400e3,
                    help="start of refined band in x, absolute coordinate "
                         "(default: 400 km, enter as metres)")
    p.add_argument("--refine-xmax", type=float, default=500e3,
                    help="end of refined band in x, absolute coordinate "
                         "(default: 500 km, enter as metres)")
    p.add_argument("--transition", type=float, default=20e3,
                    help="distance over which mesh size ramps from refined to "
                         "background outside the band (default: 20 km, enter "
                         "as metres)")
    p.add_argument("--centerline-scale", type=float, default=1.0,
                    help="scale factor applied to whatever mesh size would "
                         "otherwise apply (dx-refined or dx-background) in a "
                         "strip around y=0, for resolving centerline features "
                         "such as subglacial outflow at the grounding line. "
                         "E.g. 0.1 gives element size ten times smaller than "
                         "it would otherwise be at the centerline. Default 1 "
                         "disables this (default 1)")
    p.add_argument("--centerline-yin", type=float, default=2000.0,
                    help="half-width of the fully-scaled centerline band "
                         "(default: 2 km, enter as metres)")
    p.add_argument("--centerline-yout", type=float, default=5000.0,
                    help="half-width beyond which centerline scaling has fully "
                         "relaxed back to no effect (default: 5 km, enter as "
                         "metres)")
    p.add_argument("--output", "-o", default="mismip_plus.geo",
                    help="output .geo file path (default mismip_plus.geo)")
    p.add_argument("--plot", action="store_true",
                    help="also render a 3D bedrock visualization for this domain "
                         "(requires numpy and matplotlib)")
    p.add_argument("--plot-output",
                    help="path to save the bedrock plot (default: --output with "
                         "its extension replaced by _bedrock.png)")
    return p.parse_args()


def main():
    args = parse_args()
    x_max = ACTIVE_X_MAX
    x_min = ACTIVE_X_MAX - args.lx
    y_min = -args.ly / 2
    y_max = args.ly / 2
    if args.centerline_scale != 1.0:
        centerline_field = CENTERLINE_FIELD_TEMPLATE.format(
            centerline_scale=args.centerline_scale,
            x_min=x_min, x_max=x_max,
            centerline_yin=args.centerline_yin, centerline_yin_neg=-args.centerline_yin,
            centerline_thickness=args.centerline_yout - args.centerline_yin,
        )
        length_min = args.dx_refined * min(1.0, args.centerline_scale)
    else:
        centerline_field = NO_CENTERLINE_FIELD
        length_min = args.dx_refined

    geo = GEO_TEMPLATE.format(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        dx_background=args.dx_background, dx_refined=args.dx_refined,
        refine_xmin=args.refine_xmin, refine_xmax=args.refine_xmax,
        transition=args.transition, centerline_field=centerline_field,
        length_min=length_min,
    )
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(geo)
    print(f"wrote {args.output}")

    if args.plot:
        plot_output = args.plot_output
        if plot_output is None:
            plot_output = os.path.splitext(args.output)[0] + "_bedrock.png"
        plot_bedrock(x_min, x_max, y_min, y_max, args.refine_xmin, args.refine_xmax,
                     plot_output)


if __name__ == "__main__":
    main()
