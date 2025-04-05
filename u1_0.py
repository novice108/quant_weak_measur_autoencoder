import numpy as np
from qutip import basis, qeye, sigmax, sigmay, sigmaz, mesolve, Bloch, Options, expect
import matplotlib.pyplot as plt

def weak_measurement(psi0, H, times, strength, measurement_operator):
    """Симулює слабкі вимірювання."""
    if strength == 0.0:
        options = Options(store_states=True)
        return mesolve(H, psi0, times, [], [sigmax(), sigmay(), sigmaz()], options=options)
    else:
        M0 = np.sqrt(1 - strength) * qeye(2)
        M1 = np.sqrt(strength) * measurement_operator
        assert np.allclose((M0.dag() * M0 + M1.dag() * M1).full(), qeye(2).full()) or strength == 0.0, "Оператори не повні!"
        options = Options(store_states=True)
        return mesolve(H, psi0, times, [M0, M1], [sigmax(), sigmay(), sigmaz()], options=options)

# Початкові параметри
psi0 = basis(2, 0)
H = 0.5 * sigmax()
times = np.linspace(0, 10, 100)
measurement_operator = sigmaz()

# Симуляція зі слабким вимірюванням
strength_weak = 0.1
result_weak = weak_measurement(psi0, H, times, strength_weak, measurement_operator)

# Симуляція з сильним вимірюванням (для порівняння)
strength_strong = 1.0
result_strong = weak_measurement(psi0, H, times, strength_strong, measurement_operator)

# Візуалізація результатів
plt.figure(figsize=(12, 6))

# Очікувані значення
plt.subplot(1, 2, 1)
plt.plot(times, expect(sigmax(), result_weak.states), label=r'Слабке $\langle \sigma_x \rangle$')
plt.plot(times, expect(sigmay(), result_weak.states), label=r'Слабке $\langle \sigma_y \rangle$')
plt.plot(times, expect(sigmaz(), result_weak.states), label=r'Слабке $\langle \sigma_z \rangle$')

plt.plot(times, expect(sigmax(), result_strong.states), '--', label=r'Сильне $\langle \sigma_x \rangle$')
plt.plot(times, expect(sigmay(), result_strong.states), '--', label=r'Сильне $\langle \sigma_y \rangle$')
plt.plot(times, expect(sigmaz(), result_strong.states), '--', label=r'Сильне $\langle \sigma_z \rangle$')

plt.title("Порівняння слабкого та сильного вимірювання σz")
plt.xlabel("Час")
plt.ylabel("Очікувані значення")
plt.legend()
plt.grid()

# Сфера Блоха (для останнього стану)
plt.subplot(1, 2, 2)
if result_weak.states:
    b_weak = Bloch()
    b_weak.add_states(result_weak.states[-1])
    b_weak.zlabel = ['0', '1']
    b_weak.xlabel = ['+','-']
    b_weak.ylabel = ['+i','-i']
    b_weak.title = "Стан на сфері Блоха (слабке вимірювання)"
    b_weak.show()
else:
    b_weak = Bloch()
    b_weak.clear()
    b_weak.title = "Стани не збережено (strength = 0)"
    b_weak.show()
    print("Стани не були збережені. Strength = 0")

plt.tight_layout()
plt.show()