"""Physical camera model for `flint.camera` (sota:D).

A thin-lens camera body + lens: focal length, f-number, shutter, ISO, sensor size and focus distance.
Everything the renderer needs is derived from these six numbers, so depth of field, motion blur, exposure,
sensor noise and optical vignetting stay mutually consistent (the exposure triangle):

- Exposure value (ISO 2720 / Lagarde & de Rousiers 2014, "Moving Frostbite to PBR", s. 4.3):
  `EV100 = log2(N^2 / t * 100 / S)`.
- Scene calibration: Flint scene radiance is not in cd/m^2, so `ev100_unity` names the EV100 at which the
  exposure multiplier is exactly 1.0 (default: f/8, 1/125 s, ISO 100 = EV100 12.97, which keeps every existing
  scene looking the same). `exposure = 2^(ev100_unity - EV100 + compensation)`; this is the saturation-based
  `1 / (1.2 * 2^EV100)` scale up to the constant scene-to-photometric factor.
- Thin-lens circle of confusion (Potmesil & Chakravarty 1981): diameter on the sensor
  `c(z) = A * f * |z - S| / (z * (S - f))` with aperture diameter `A = f / N`.
- Field of view from the sensor height and the image distance.
- Auto exposure: given the metered EV100 the renderer's histogram produced, `solve_exposure_triangle` picks
  shutter (aperture priority) and then ISO (auto-ISO) the way a camera body does, so the motion-blur shutter and
  the grain follow the light level.
- Sensor noise: photon shot noise + read noise in exposure-normalised units (`sensor_noise_sigma`).

Pure math, no renderer or physics imports, so it can be used by `CameraRig`, tools and tests alike.
"""
from std.math import atan, exp2, log2, max, min, sqrt, tan

comptime PHYS_EXPOSURE_MANUAL: Int = 0
comptime PHYS_EXPOSURE_APERTURE_PRIORITY: Int = 1   # N fixed; t (then ISO) follow the meter
comptime PHYS_EXPOSURE_SHUTTER_PRIORITY: Int = 2    # t fixed; N (then ISO) follow the meter
comptime PHYS_EXPOSURE_PROGRAM: Int = 3             # N and t from a program line; ISO fixed unless out of range

comptime PHYS_EV100_UNITY_DEFAULT: Float32 = 12.965784  # log2(8^2 / (1/125)) at ISO 100


@fieldwise_init
struct ExposureTriangle(Copyable, ImplicitlyCopyable, Movable):
    """Result of a metering solve: the settings a camera body would pick."""
    var fstop: Float32
    var shutter_s: Float32
    var iso: Float32
    var ev100: Float32


struct PhysicalCameraModel(Copyable, ImplicitlyCopyable, Movable):
    """Thin-lens camera body + lens. Units: millimetres for the lens/sensor, metres for scene distances."""
    var focal_length_mm: Float32
    var fstop: Float32
    var max_aperture_fstop: Float32  # widest stop the lens has (optical vignetting scales off this)
    var shutter_s: Float32
    var iso: Float32
    var sensor_width_mm: Float32
    var sensor_height_mm: Float32
    var focus_distance_m: Float32
    var aperture_blades: Int
    var blade_rotation_deg: Float32
    var blade_curvature: Float32      # 0 = straight polygon blades, 1 = circular iris
    var exposure_mode: Int
    var exposure_compensation_ev: Float32
    var ev100_unity: Float32
    var min_shutter_s: Float32        # slowest hand-holdable shutter before auto-ISO kicks in
    var max_iso: Float32
    var full_well_photons: Float32    # photons that saturate a pixel at ISO 100 (shot-noise scale)
    var read_noise_photons: Float32

    def __init__(
        out self,
        focal_length_mm: Float32 = 50.0,
        fstop: Float32 = 8.0,
        shutter_s: Float32 = 1.0 / 125.0,
        iso: Float32 = 100.0,
        focus_distance_m: Float32 = 5.0,
        sensor_width_mm: Float32 = 36.0,
        sensor_height_mm: Float32 = 24.0,
        aperture_blades: Int = 7,
    ):
        self.focal_length_mm = max(Float32(4.0), focal_length_mm)
        self.fstop = max(Float32(0.7), fstop)
        self.max_aperture_fstop = min(Float32(1.4), self.fstop)
        self.shutter_s = max(Float32(1.0e-5), shutter_s)
        self.iso = max(Float32(25.0), iso)
        self.sensor_width_mm = max(Float32(1.0), sensor_width_mm)
        self.sensor_height_mm = max(Float32(1.0), sensor_height_mm)
        self.focus_distance_m = max(Float32(0.05), focus_distance_m)
        self.aperture_blades = max(0, min(16, aperture_blades))
        self.blade_rotation_deg = Float32(0.0)
        self.blade_curvature = Float32(0.0)
        self.exposure_mode = PHYS_EXPOSURE_MANUAL
        self.exposure_compensation_ev = Float32(0.0)
        self.ev100_unity = PHYS_EV100_UNITY_DEFAULT
        self.min_shutter_s = Float32(1.0 / 60.0)
        self.max_iso = Float32(25600.0)
        self.full_well_photons = Float32(24000.0)
        self.read_noise_photons = Float32(3.0)

    # ------------------------------------------------------------------ exposure
    def ev100(self) -> Float32:
        """ISO 2720 exposure value normalised to ISO 100."""
        return log2(max(Float32(1.0e-12), (self.fstop * self.fstop / self.shutter_s) * (Float32(100.0) / self.iso)))

    def exposure_scale(self) -> Float32:
        """Multiplier applied to scene radiance before the tone curve (`ren.exposure`)."""
        return exp2(self.ev100_unity - self.ev100() + self.exposure_compensation_ev)

    def ev100_for_exposure(self, exposure: Float32) -> Float32:
        """Inverse of `exposure_scale`: the EV100 a total exposure multiplier corresponds to."""
        return self.ev100_unity + self.exposure_compensation_ev - log2(max(Float32(1.0e-12), exposure))

    def solve_exposure_triangle(self, metered_ev100: Float32) -> ExposureTriangle:
        """Settings a body in `exposure_mode` picks for a scene metered at `metered_ev100`.
        Aperture priority keeps N, picks t = N^2 * 100 / (S * 2^EV); if t would be slower than
        `min_shutter_s`, t is pinned there and ISO rises (auto-ISO) up to `max_iso`.
        Shutter priority keeps t and solves N within [max_aperture, 22]; ISO covers the rest.
        Program mode follows a simple line `N = sqrt(2^EV * t)` with `t = 1 / focal_length`."""
        var ev = metered_ev100
        var n = self.fstop
        var t = self.shutter_s
        var s = self.iso
        if self.exposure_mode == PHYS_EXPOSURE_MANUAL:
            return ExposureTriangle(n, t, s, self.ev100())
        if self.exposure_mode == PHYS_EXPOSURE_SHUTTER_PRIORITY:
            s = Float32(100.0)
            n = sqrt(max(Float32(1.0e-6), exp2(ev) * t))
            if n < self.max_aperture_fstop:
                n = self.max_aperture_fstop
                s = Float32(100.0) * (n * n / t) / exp2(ev)
            n = min(n, Float32(22.0))
        elif self.exposure_mode == PHYS_EXPOSURE_PROGRAM:
            s = Float32(100.0)
            t = min(Float32(1.0) / max(Float32(1.0), self.focal_length_mm), Float32(1.0) / Float32(60.0))
            n = sqrt(max(Float32(1.0e-6), exp2(ev) * t))
            if n < self.max_aperture_fstop:
                n = self.max_aperture_fstop
                t = n * n / exp2(ev)
                if t > self.min_shutter_s:
                    t = self.min_shutter_s
                    s = Float32(100.0) * (n * n / t) / exp2(ev)
            n = min(n, Float32(22.0))
        else:
            s = Float32(100.0)
            t = (n * n) * Float32(100.0) / (s * exp2(ev))
            if t > self.min_shutter_s:
                t = self.min_shutter_s
                s = Float32(100.0) * (n * n / t) / exp2(ev)
        s = max(Float32(25.0), min(self.max_iso, s))
        t = max(Float32(1.0e-5), t)
        var out_ev = log2((n * n / t) * (Float32(100.0) / s))
        return ExposureTriangle(n, t, s, out_ev)

    def apply_exposure_triangle(mut self, tri: ExposureTriangle):
        self.fstop = tri.fstop
        self.shutter_s = tri.shutter_s
        self.iso = tri.iso

    def shutter_fraction(self, frame_dt_s: Float32) -> Float32:
        """Fraction of the frame interval the shutter is open (`0.5` = 180-degree rotary shutter)."""
        return max(Float32(0.0), min(Float32(1.0), self.shutter_s / max(Float32(1.0e-4), frame_dt_s)))

    def shutter_angle_deg(self, frame_dt_s: Float32) -> Float32:
        return self.shutter_fraction(frame_dt_s) * Float32(360.0)

    def sensor_noise_sigma(self, signal: Float32) -> Float32:
        """Std-dev of sensor noise for an exposure-normalised signal (`1.0` = clipping white).
        Photons collected at this ISO: `Q = full_well * 100 / ISO` per unit signal; shot noise is
        Poisson (`sqrt(signal * Q)`), read noise adds in quadrature; dividing by `Q` returns to signal units."""
        var q = self.full_well_photons * Float32(100.0) / self.iso
        var var_ph = max(Float32(0.0), signal) * q + self.read_noise_photons * self.read_noise_photons
        return sqrt(var_ph) / q

    # ------------------------------------------------------------------ geometry
    def image_distance_mm(self) -> Float32:
        """Lens-to-sensor distance when focused at `focus_distance_m` (thin-lens equation)."""
        var f = self.focal_length_mm
        var s_mm = max(self.focus_distance_m * Float32(1000.0), f * Float32(1.001))
        return f * s_mm / (s_mm - f)

    def fov_y_deg(self, account_for_focus: Bool = False) -> Float32:
        var d = self.image_distance_mm() if account_for_focus else self.focal_length_mm
        return Float32(2.0) * atan(self.sensor_height_mm / (Float32(2.0) * d)) * Float32(57.29577951)

    def fov_x_deg(self) -> Float32:
        return Float32(2.0) * atan(self.sensor_width_mm / (Float32(2.0) * self.focal_length_mm)) * Float32(57.29577951)

    def set_focal_length_for_fov_y(mut self, fov_y_deg: Float32):
        var half = max(Float32(0.5), min(Float32(170.0), fov_y_deg)) * Float32(0.5) * Float32(0.017453293)
        self.focal_length_mm = self.sensor_height_mm / (Float32(2.0) * tan(half))

    def aperture_diameter_mm(self) -> Float32:
        return self.focal_length_mm / self.fstop

    def coc_diameter_mm(self, z_m: Float32) -> Float32:
        """Signed thin-lens circle-of-confusion diameter on the sensor (`> 0` behind the focus plane)."""
        var f = self.focal_length_mm * Float32(0.001)
        var s = max(self.focus_distance_m, f * Float32(1.001))
        var z = max(Float32(1.0e-4), z_m)
        var a = f / self.fstop
        return a * f * (z - s) / (z * (s - f)) * Float32(1000.0)

    def coc_radius_px(self, z_m: Float32, image_height_px: Int) -> Float32:
        """Signed CoC radius in output pixels (what the DoF passes consume)."""
        return Float32(0.5) * self.coc_diameter_mm(z_m) / self.sensor_height_mm * Float32(image_height_px)

    def coc_infinity_radius_px(self, image_height_px: Int) -> Float32:
        """`K` in `coc_px(z) = K * (1 - S / z)`: the CoC radius of an object at infinity."""
        var f = self.focal_length_mm * Float32(0.001)
        var s = max(self.focus_distance_m, f * Float32(1.001))
        var c_inf_mm = (f * f) / (self.fstop * (s - f)) * Float32(1000.0)
        return Float32(0.5) * c_inf_mm / self.sensor_height_mm * Float32(image_height_px)

    def acceptable_coc_mm(self) -> Float32:
        """Zeiss criterion: sensor diagonal / 1500 (0.029 mm on full frame)."""
        return sqrt(self.sensor_width_mm * self.sensor_width_mm + self.sensor_height_mm * self.sensor_height_mm) / Float32(1500.0)

    def hyperfocal_distance_m(self) -> Float32:
        var f = self.focal_length_mm
        return (f * f / (self.fstop * self.acceptable_coc_mm()) + f) * Float32(0.001)

    def depth_of_field_limits_m(self) -> Tuple[Float32, Float32]:
        """Near / far limits of acceptable sharpness; far is `1e9` at or beyond the hyperfocal distance."""
        var h = self.hyperfocal_distance_m()
        var s = self.focus_distance_m
        var f = self.focal_length_mm * Float32(0.001)
        var near = s * (h - f) / (h + s - Float32(2.0) * f)
        var far = Float32(1.0e9)
        if s < h:
            far = s * (h - f) / (h - s)
        return (near, far)

    def optical_vignette_strength(self) -> Float32:
        """Cat's-eye / optical vignetting fades as the lens is stopped down; gone ~2 stops below wide open."""
        var stops_down = log2(max(Float32(1.0), self.fstop / self.max_aperture_fstop)) * Float32(2.0)
        return max(Float32(0.0), Float32(1.0) - stops_down * Float32(0.5))
