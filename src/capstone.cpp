#include "capstone.h"
#include "utils.h"

constexpr double ACCEL_STD = 1.0;
constexpr double GYRO_STD = 0.01 / 180.0 * M_PI;
constexpr double INIT_VEL_STD = 10.0;
constexpr double INIT_PSI_STD = 45.0 / 180.0 * M_PI;
constexpr double INIT_BIAS_STD = 0.01 / 180.0 * M_PI;
constexpr double GYRO_BIAS_RATE = 0.01 / 180.0 * M_PI;
constexpr double GPS_POS_STD = 3.0;
constexpr double LIDAR_RANGE_STD = 3.0;
constexpr double LIDAR_THETA_STD = 0.02;
constexpr bool INIT_ON_LIDAR_MEASUREMENTS = true;
constexpr bool INIT_ON_GPS_MEASUREMENTS = false;
constexpr double CHI_SQUARE_P_VALUE = 0.95;

static int count = 0;
std::vector<GPSMeasurement> gps_measurements;

bool chiSquareTest(const VectorXd &innovation, const MatrixXd &S, int dof) {
    boost::math::chi_squared dist(dof);
    double chi_square_threshold = quantile(dist, CHI_SQUARE_P_VALUE);
    double nis = innovation.transpose() * S.inverse() * innovation;
    return (nis < chi_square_threshold);
}

VectorXd normaliseState(VectorXd state) {
    state(2) = wrapAngle(state(2));
    return state;
}

VectorXd normaliseLidarMeasurement(VectorXd meas) {
    meas(1) = wrapAngle(meas(1));
    return meas;
}


std::vector<VectorXd> generateSigmaPoints(VectorXd state, MatrixXd covariance) {
    std::vector<VectorXd> sigmaPoints;
    int n = state.size();
    double k = 3.0 - n;
    MatrixXd sqrtCovariance = covariance.llt().matrixL();

    sigmaPoints.push_back(state);

    for (unsigned int i = 0; i < n; i++) {
        sigmaPoints.emplace_back(state + sqrt(n + k) * sqrtCovariance.col(i));
        sigmaPoints.emplace_back(state - sqrt(n + k) * sqrtCovariance.col(i));
    }

    return sigmaPoints;
}

std::vector<double> generateSigmaWeights(unsigned int numStates) {
    std::vector<double> sigmaWeights;
    double k = 3.0 - numStates;

    sigmaWeights.push_back(k / (numStates + k));
    double weight = 0.5 / (numStates + k);
    for (unsigned int i = 0; i < 2 * numStates; i++) {
        sigmaWeights.push_back(weight);
    }

    return sigmaWeights;
}


VectorXd vehicleProcessModel(VectorXd aug_state, double psi_dot, double dt) {
    VectorXd new_state = VectorXd::Zero(5);

    double old_px = aug_state(0);
    double old_py = aug_state(1);
    double old_psi = aug_state(2);
    double old_v = aug_state(3);
    double old_gyro_bias = aug_state(4);
    double psi_std = aug_state(5);
    double acc_std = aug_state(6);


    double new_px = old_px + dt * old_v * cos(old_psi);
    double new_py = old_py + dt * old_v * sin(old_psi);
    double new_psi = old_psi + dt * (psi_dot + psi_std - old_gyro_bias);
    double new_v = old_v + dt * acc_std;
    double new_gyro_bias = old_gyro_bias;

    new_state <<
            new_px,
            new_py,
            new_psi,
            new_v,
            new_gyro_bias;

    return new_state;
}

VectorXd lidarMeasurementModel(VectorXd aug_state, double beaconX, double beaconY) {
    VectorXd z_hat = VectorXd::Zero(2);

    double delta_x = beaconX - aug_state(0);
    double delta_y = beaconY - aug_state(1);

    z_hat[0] = sqrt(pow(delta_x, 2) + pow(delta_y, 2)) + aug_state(5);
    z_hat[1] = atan2(delta_y, delta_x) - aug_state(2) + aug_state(6);

    return z_hat;
}

void KalmanFilter::handleLidarMeasurement(LidarMeasurement meas, const BeaconMap &map) {
    if (isInitialised()) {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        BeaconData map_beacon = map.getBeaconWithId(meas.id); // Match Beacon with built in Data Association Id
        if (meas.id != -1 && map_beacon.id != -1) // Check that we have a valid beacon match
        {
            VectorXd z = Vector2d::Zero();
            z << meas.range, meas.theta;

            MatrixXd R = Matrix2d::Zero();
            R(0, 0) = LIDAR_RANGE_STD * LIDAR_RANGE_STD;
            R(1, 1) = LIDAR_THETA_STD * LIDAR_THETA_STD;

            int n = state.size();
            int w = 2;

            VectorXd state_aug = VectorXd::Zero(n + w);
            state_aug.head(n) = state;

            MatrixXd cov_aug = MatrixXd::Zero(n + w, n + w);
            cov_aug.topLeftCorner(n, n) = cov;
            cov_aug.bottomRightCorner(w, w) = R;

            auto sigma_points = generateSigmaPoints(state_aug, cov_aug);
            auto sigma_weights = generateSigmaWeights(n + w);

            std::vector<VectorXd> z_sigma_points;
            for (const auto &point: sigma_points) {
                z_sigma_points.push_back(lidarMeasurementModel(point, map_beacon.x, map_beacon.y));
            }

            VectorXd z_mean = Vector2d::Zero();
            for (unsigned int i = 0; i < z_sigma_points.size(); i++) {
                z_mean += sigma_weights[i] * z_sigma_points[i];
            }

            VectorXd y = normaliseLidarMeasurement(z - z_mean);

            MatrixXd S = MatrixXd::Zero(2, 2);
            for (unsigned int i = 0; i < z_sigma_points.size(); i++) {
                VectorXd diff = normaliseLidarMeasurement(z_sigma_points[i] - z_mean);
                S += sigma_weights[i] * diff * diff.transpose();
            }

            MatrixXd Pxy = MatrixXd::Zero(n, 2);
            for (unsigned int i = 0; i < z_sigma_points.size(); i++) {
                VectorXd x_diff = normaliseState(sigma_points[i].head(n) - state);
                VectorXd z_diff = normaliseLidarMeasurement(z_sigma_points[i] - z_mean);
                Pxy += sigma_weights[i] * x_diff * z_diff.transpose();
            }

            MatrixXd K = Pxy * S.inverse();
            state = state + K * y;
            cov = cov - K * S * K.transpose();
        }

        setState(state);
        setCovariance(cov);
    }
}

void KalmanFilter::predictionStep(GyroMeasurement gyro, double dt) {
    if (isInitialised()) {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        int n = state.size();
        int w = 3;

        MatrixXd Q = MatrixXd::Zero(w, w);
        Q(0, 0) = GYRO_STD * GYRO_STD;
        Q(1, 1) = ACCEL_STD * ACCEL_STD;
        Q(2, 2) = GYRO_BIAS_RATE * GYRO_BIAS_RATE;

        VectorXd aug_state = VectorXd::Zero(n + w);
        aug_state.head(n) = state;

        MatrixXd aug_cov = MatrixXd::Zero(n + w, n + w);
        aug_cov.topLeftCorner(n, n) = cov;
        aug_cov.bottomRightCorner(w, w) = Q;

        std::vector<VectorXd> sigma_points = generateSigmaPoints(aug_state, aug_cov);
        std::vector<double> sigma_weights = generateSigmaWeights(n + w);


        std::vector<VectorXd> sigma_point_predictions;
        for (const auto &sigma_point: sigma_points) {
            sigma_point_predictions.push_back(vehicleProcessModel(sigma_point, gyro.psi_dot, dt));
        }

        state = VectorXd::Zero(n);
        for (unsigned int i = 0; i < sigma_point_predictions.size(); i++) {
            state += sigma_weights[i] * sigma_point_predictions[i];
        }
        state = normaliseState(state);

        cov = MatrixXd::Zero(n, n);
        for (unsigned int i = 0; i < sigma_point_predictions.size(); i++) {
            VectorXd diff = normaliseState(sigma_point_predictions[i] - state);
            cov += sigma_weights[i] * diff * diff.transpose();
        }

        setState(state);
        setCovariance(cov);
    }
}

void KalmanFilter::handleGPSMeasurement(GPSMeasurement meas) {
    if (isInitialised()) {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        VectorXd z = Vector2d::Zero();
        MatrixXd H = MatrixXd(2, 5);
        MatrixXd R = Matrix2d::Zero();

        z <<
                meas.x,
                meas.y;
        H <<
                1, 0, 0, 0, 0,
                0, 1, 0, 0, 0;

        R(0, 0) = GPS_POS_STD * GPS_POS_STD;
        R(1, 1) = GPS_POS_STD * GPS_POS_STD;

        VectorXd z_hat = H * state;
        VectorXd y = z - z_hat;
        MatrixXd S = H * cov * H.transpose() + R;

        std::cout << "New GPS Measurement: " + std::to_string(++count) << std::endl;
        if (!chiSquareTest(y, S, z.size())) return;

        std::cout << "GPS Measurement Passed Chi-Square Test: " + std::to_string(count) << std::endl;

        MatrixXd K = cov * H.transpose() * S.inverse();

        state = state + K * y;
        cov = (MatrixXd::Identity(5, 5) - K * H) * cov;

        setState(state);
        setCovariance(cov);
    } else if (INIT_ON_LIDAR_MEASUREMENTS) {
        // Initialise on Lidar Measurements
        if (gps_measurements.size() == 3) gps_measurements.erase(gps_measurements.begin());
        gps_measurements.push_back(meas);
    } else if (INIT_ON_GPS_MEASUREMENTS) {
        // Initialise on Two GPS measurements
        if (gps_measurements.size() == 3) {
            VectorXd state = VectorXd::Zero(5);
            state(0) = meas.x;
            state(1) = meas.y;
            state(2) = atan2(gps_measurements.back().y - meas.y, gps_measurements.back().x - meas.x);
            state(4) = GYRO_BIAS_RATE;

            state = normaliseState(state);

            MatrixXd cov = MatrixXd::Zero(5, 5);
            cov(0, 0) = GPS_POS_STD * GPS_POS_STD;
            cov(1, 1) = GPS_POS_STD * GPS_POS_STD;
            cov(2, 2) = INIT_PSI_STD * INIT_PSI_STD;
            cov(3, 3) = INIT_VEL_STD * INIT_VEL_STD;
            cov(4, 4) = INIT_BIAS_STD * INIT_BIAS_STD;

            setState(state);
            setCovariance(cov);
            setInitialised(true);
        } else {
            gps_measurements.push_back(meas);
        }
    } else {
        // Initialise on first GPS measurement
        VectorXd state = VectorXd::Zero(5);
        state(0) = meas.x;
        state(1) = meas.y;
        state(4) = GYRO_BIAS_RATE;

        MatrixXd cov = MatrixXd::Zero(5, 5);
        cov(0, 0) = GPS_POS_STD * GPS_POS_STD;
        cov(1, 1) = GPS_POS_STD * GPS_POS_STD;
        cov(2, 2) = INIT_PSI_STD * INIT_PSI_STD;
        cov(3, 3) = INIT_VEL_STD * INIT_VEL_STD;
        cov(4, 4) = INIT_BIAS_STD * INIT_BIAS_STD;

        setState(state);
        setCovariance(cov);
        setInitialised(true);
    }
}

void KalmanFilter::handleLidarMeasurements(const std::vector<LidarMeasurement> &dataset, const BeaconMap &map) {
    if (!isInitialised() && dataset.size() >= 5 && gps_measurements.size() > 2) {
        std::vector<double> heading_estimates;
        GPSMeasurement gps_prev = gps_measurements.at(gps_measurements.size() - 2);
        GPSMeasurement gps_curr = gps_measurements.back();
        double vehicle_x = gps_curr.x;
        double vehicle_y = gps_curr.y;

        for (const auto &meas: dataset) {
            BeaconData map_beacon = map.getBeaconWithId(meas.id);
            if (meas.id != -1 && map_beacon.id != -1) {
                double global_bearing = atan2(map_beacon.y - vehicle_y, map_beacon.x - vehicle_x);
                double heading_estimate = global_bearing - meas.theta;
                heading_estimate = wrapAngle(heading_estimate);
                heading_estimates.push_back(heading_estimate);
            }
        }

        if (heading_estimates.size() >= 5) {
            std::sort(heading_estimates.begin(), heading_estimates.end());

            double median_heading = heading_estimates[heading_estimates.size() / 2];

            std::vector<double> filtered_headings;
            for (const auto &heading: heading_estimates) {
                double diff = wrapAngle(heading - median_heading);
                if (std::fabs(diff) < M_PI / 4) {
                    filtered_headings.push_back(heading);
                }
            }

            double sin_sum = 0.0;
            double cos_sum = 0.0;
            for (const auto &heading: filtered_headings) {
                sin_sum += sin(heading);
                cos_sum += cos(heading);
            }

            double lidar_heading = atan2(sin_sum, cos_sum);
            double gps_heading = atan2(gps_prev.y - gps_curr.y, gps_prev.x - gps_curr.x);

            double diff = fabs(wrapAngle(lidar_heading - gps_heading));
            double final_heading = (diff < M_PI / 4) ? wrapAngle(lidar_heading) : wrapAngle(gps_heading + M_PI);

            VectorXd state = VectorXd::Zero(5);
            MatrixXd cov = MatrixXd::Zero(5, 5);

            state(0) = gps_curr.x;
            state(1) = gps_curr.y;
            state(2) = final_heading;
            state(4) = GYRO_BIAS_RATE;

            cov(0, 0) = GPS_POS_STD * GPS_POS_STD;
            cov(1, 1) = GPS_POS_STD * GPS_POS_STD;
            cov(2, 2) = INIT_PSI_STD * INIT_PSI_STD;
            cov(3, 3) = INIT_VEL_STD * INIT_VEL_STD;
            cov(4, 4) = INIT_BIAS_STD * INIT_BIAS_STD;

            setState(state);
            setCovariance(cov);
            setInitialised(true);

            for (const auto &meas: dataset) {
                handleLidarMeasurement(meas, map);
            }
        }
    }

    if (isInitialised()) {
        for (const auto &meas: dataset) {
            handleLidarMeasurement(meas, map);
        }
    }
}

Matrix2d KalmanFilter::getVehicleStatePositionCovariance() {
    Matrix2d pos_cov = Matrix2d::Zero();
    MatrixXd cov = getCovariance();
    if (isInitialised() && cov.size() != 0) {
        pos_cov << cov(0, 0), cov(0, 1), cov(1, 0), cov(1, 1);
    }
    return pos_cov;
}

VehicleState KalmanFilter::getVehicleState() {
    if (isInitialised()) {
        VectorXd state = getState(); // STATE VECTOR [X,Y,PSI,V,...]
        return VehicleState(state[0], state[1], state[2], state[3]);
    }
    return VehicleState();
}

void KalmanFilter::predictionStep(double dt) {
}
